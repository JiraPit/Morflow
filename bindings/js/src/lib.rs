#[macro_use]
extern crate napi_derive;

use std::path::Path;

use abi_stable::std_types::RVec;
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use napi::bindgen_prelude::*;
use napi::Env;
use pipeline::{Morflow, MorflowError, MorflowPipeline};

fn map_error(err: MorflowError) -> napi::Error {
    match err {
        MorflowError::Io(e) => {
            napi::Error::new(napi::Status::GenericFailure, format!("IO error: {}", e))
        }
        MorflowError::Parse(e) => {
            napi::Error::new(napi::Status::InvalidArg, format!("Parse error: {}", e))
        }
        MorflowError::Compile(e) => {
            napi::Error::new(napi::Status::InvalidArg, format!("Compile error: {}", e))
        }
        MorflowError::Action(e) => {
            napi::Error::new(napi::Status::GenericFailure, format!("Action error: {}", e))
        }
        MorflowError::Execution(e) => napi::Error::new(
            napi::Status::GenericFailure,
            format!("Execution error: {}", e),
        ),
        MorflowError::TypeMismatch(e) => {
            napi::Error::new(napi::Status::InvalidArg, format!("Type error: {}", e))
        }
    }
}

/// A structured multi-dimensional tensor representation in JavaScript/TypeScript.
#[napi(object)]
pub struct MorflowTensor {
    /// Shape array of dimensions (e.g. [H, W, C] or [channels, samples])
    pub shape: Vec<u32>,
    /// Data type: 'f32', 'u8', 'i32', or 'raw'
    pub dtype: String,
    /// Raw binary data buffer
    pub data: Buffer,
}

/// Helper input struct when passing custom shape along with typed array buffer
#[napi(object)]
pub struct TensorInput {
    pub data: Buffer,
    pub shape: Vec<u32>,
    pub dtype: Option<String>,
}

fn tensor_to_morflow_tensor(tensor: &Tensor) -> MorflowTensor {
    let shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();
    let dtype = match tensor.dtype {
        TensorDType::F32 => "f32",
        TensorDType::U8 => "u8",
        TensorDType::I32 => "i32",
        _ => "raw",
    }
    .to_string();

    let data = if let Some(bytes) = tensor.as_bytes() {
        Buffer::from(bytes)
    } else {
        Buffer::from(tensor.to_contiguous_bytes().as_slice())
    };

    MorflowTensor { shape, dtype, data }
}

fn payload_to_morflow_tensor(payload: Payload) -> napi::Result<MorflowTensor> {
    match payload {
        Payload::Tensor(t) => Ok(tensor_to_morflow_tensor(&t)),
        Payload::Image(img) => Ok(tensor_to_morflow_tensor(&img.tensor)),
        Payload::Audio(aud) => Ok(tensor_to_morflow_tensor(&aud.tensor)),
        Payload::Data { buffer } => Ok(MorflowTensor {
            shape: vec![buffer.len() as u32],
            dtype: "u8".to_string(),
            data: Buffer::from(buffer.as_slice()),
        }),
        Payload::WithArgs { payload, .. } => {
            payload_to_morflow_tensor(abi_stable::std_types::RBox::into_inner(payload))
        }
        Payload::Error(err) => Err(napi::Error::new(
            napi::Status::GenericFailure,
            err.to_string(),
        )),
        Payload::Composite(_) => Err(napi::Error::new(
            napi::Status::GenericFailure,
            "Composite payload returned where single tensor was expected",
        )),
    }
}

fn tensor_input_to_payload(input: TensorInput) -> napi::Result<Payload> {
    let shape: Vec<usize> = input.shape.iter().map(|&d| d as usize).collect();
    let dtype_str = input.dtype.unwrap_or_else(|| "f32".to_string());
    let bytes = input.data.as_ref();

    let tensor = match dtype_str.to_lowercase().as_str() {
        "u8" | "uint8" | "byte" => {
            Tensor::from_rvec_u8(RVec::from(bytes.to_vec()), shape, TensorDType::U8)
                .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))?
        }
        "i32" | "int32" => {
            Tensor::from_rvec_u8(RVec::from(bytes.to_vec()), shape, TensorDType::I32)
                .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))?
        }
        _ /* "f32" */ => {
            if bytes.len() % 4 != 0 {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    "Data buffer size is not a multiple of 4 for Float32 tensor",
                ));
            }
            let f32_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            Tensor::from_f32_shape(f32_slice, shape)
                .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))?
        }
    };

    if tensor.rank() == 3 {
        let channels = tensor.shape[2];
        let cs = match channels {
            1 => Some(ColorSpace::Grayscale),
            3 => Some(ColorSpace::Rgb),
            4 => Some(ColorSpace::Rgba),
            _ => None,
        };
        if let Some(color_space) = cs {
            let img = Image::new(tensor, color_space, ImageLayout::Hwc)
                .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))?;
            return Ok(Payload::Image(img));
        }
    } else if tensor.rank() == 2 && tensor.shape[0] <= 8 {
        let ch = tensor.shape[0];
        let layout = AudioChannelLayout::from_channel_count(ch);
        let aud = Audio::new(tensor, 44100, layout, AudioLayout::Planar)
            .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))?;
        return Ok(Payload::Audio(aud));
    }
    Ok(Payload::Tensor(tensor))
}

fn float32_array_to_payload(f32_arr: Float32Array) -> napi::Result<Payload> {
    let slice = f32_arr.as_ref();
    let shape = vec![slice.len()];
    let tensor = Tensor::from_f32_shape(slice, shape)
        .map_err(|e| napi::Error::new(napi::Status::InvalidArg, e.to_string()))?;
    Ok(Payload::Tensor(tensor))
}

fn buffer_to_payload(buf: Buffer) -> Payload {
    let bytes = buf.as_ref();
    Payload::Data {
        buffer: RVec::from(bytes.to_vec()),
    }
}

fn extract_input_payload(
    input: Option<Either3<TensorInput, Float32Array, Buffer>>,
) -> napi::Result<Payload> {
    match input {
        Some(Either3::A(tensor_in)) => tensor_input_to_payload(tensor_in),
        Some(Either3::B(f32_arr)) => float32_array_to_payload(f32_arr),
        Some(Either3::C(buf)) => Ok(buffer_to_payload(buf)),
        None => Ok(Payload::Data {
            buffer: RVec::new(),
        }),
    }
}

/// Libuv worker task for asynchronous pipeline execution
pub struct AsyncRunTask {
    pipeline: MorflowPipeline,
    payload: Payload,
}

impl Task for AsyncRunTask {
    type Output = MorflowTensor;
    type JsValue = MorflowTensor;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let payload = std::mem::replace(
            &mut self.payload,
            Payload::Data {
                buffer: RVec::new(),
            },
        );
        let outputs = self.pipeline.run(payload).map_err(map_error)?;
        let single = outputs.into_single().map_err(map_error)?;
        payload_to_morflow_tensor(single)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

/// Libuv worker task for asynchronous multi-output pipeline execution
pub struct AsyncRunAllTask {
    pipeline: MorflowPipeline,
    payload: Payload,
}

impl Task for AsyncRunAllTask {
    type Output = Vec<(String, MorflowTensor)>;
    type JsValue = napi::JsObject;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let payload = std::mem::replace(
            &mut self.payload,
            Payload::Data {
                buffer: RVec::new(),
            },
        );
        let outputs = self.pipeline.run(payload).map_err(map_error)?;
        let mut list = Vec::with_capacity(outputs.len());
        for (name, out_payload) in outputs.into_iter() {
            let tensor_obj = payload_to_morflow_tensor(out_payload)?;
            list.push((name, tensor_obj));
        }
        Ok(list)
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        let mut obj = env.create_object()?;
        for (name, tensor) in output {
            obj.set_named_property(&name, tensor)?;
        }
        Ok(obj)
    }
}

/// An executable Morflow pipeline instance for JavaScript/TypeScript.
#[napi]
pub struct Pipeline {
    inner: MorflowPipeline,
}

#[napi]
impl Pipeline {
    /// Parameter names declared with `accept $param`.
    #[napi(getter)]
    pub fn params(&self) -> Vec<String> {
        self.inner.params().iter().map(|p| p.name.clone()).collect()
    }

    /// Preloads and warms up all declared actions in memory.
    #[napi]
    pub fn warmup(&self) -> napi::Result<()> {
        self.inner.warmup().map_err(map_error)
    }

    /// Executes the pipeline synchronously with a TensorInput, Float32Array, or Buffer.
    #[napi]
    pub fn run_sync(
        &mut self,
        input: Option<Either3<TensorInput, Float32Array, Buffer>>,
    ) -> napi::Result<MorflowTensor> {
        let payload = extract_input_payload(input)?;
        let outputs = self.inner.run(payload).map_err(map_error)?;
        let single = outputs.into_single().map_err(map_error)?;
        payload_to_morflow_tensor(single)
    }

    /// Executes the pipeline synchronously returning a map of all named output streams.
    #[napi]
    pub fn run_sync_all(
        &mut self,
        env: Env,
        input: Option<Either3<TensorInput, Float32Array, Buffer>>,
    ) -> napi::Result<napi::JsObject> {
        let payload = extract_input_payload(input)?;
        let outputs = self.inner.run(payload).map_err(map_error)?;
        let mut obj = env.create_object()?;
        for (name, out_payload) in outputs.into_iter() {
            let tensor_obj = payload_to_morflow_tensor(out_payload)?;
            obj.set_named_property(&name, tensor_obj)?;
        }
        Ok(obj)
    }

    /// Executes the pipeline asynchronously on a worker thread, returning a Promise<MorflowTensor>.
    #[napi]
    pub fn run(
        &self,
        input: Option<Either3<TensorInput, Float32Array, Buffer>>,
    ) -> napi::Result<AsyncTask<AsyncRunTask>> {
        let payload = extract_input_payload(input)?;
        Ok(AsyncTask::new(AsyncRunTask {
            pipeline: self.inner.clone(),
            payload,
        }))
    }

    /// Executes the pipeline asynchronously on a worker thread, returning a Promise<Record<string, MorflowTensor>>.
    #[napi]
    pub fn run_all(
        &self,
        input: Option<Either3<TensorInput, Float32Array, Buffer>>,
    ) -> napi::Result<AsyncTask<AsyncRunAllTask>> {
        let payload = extract_input_payload(input)?;
        Ok(AsyncTask::new(AsyncRunAllTask {
            pipeline: self.inner.clone(),
            payload,
        }))
    }
}

/// Compiles and loads a `.morf` pipeline file from disk.
#[napi]
pub fn load(path: String) -> napi::Result<Pipeline> {
    let p = Path::new(&path);
    let pipeline = Morflow::load(p).map_err(map_error)?;
    Ok(Pipeline { inner: pipeline })
}

/// Compiles a `.morf` pipeline DSL source string directly.
#[napi]
pub fn from_str(source: String) -> napi::Result<Pipeline> {
    let pipeline = Morflow::from_str(&source).map_err(map_error)?;
    Ok(Pipeline { inner: pipeline })
}
