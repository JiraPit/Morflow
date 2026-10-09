#[macro_use]
extern crate napi_derive;

use std::path::Path;

use abi_stable::std_types::RVec;
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use morflow::{Morflow, MorflowError, MorflowPipeline};
use napi::bindgen_prelude::*;
use napi::Env;

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
    /// The payload type to send. One of `"tensor"`, `"image"`, or `"audio"`.
    /// When omitted the input is a plain `"tensor"`. There is no type
    /// inference: a rank-3 array is a tensor unless you ask for `"image"`.
    pub payload_type: Option<String>,
    /// Color space for `payloadType: "image"`. Inferred from the channel count
    /// when omitted.
    pub color_space: Option<String>,
    /// Sample rate in Hz for `payloadType: "audio"`. Defaults to 44100.
    pub sample_rate: Option<u32>,
    /// Channel count for `payloadType: "audio"`. Defaults to the leading dimension.
    pub channels: Option<u32>,
    /// Memory layout: `"hwc"` / `"chw"` for images, `"planar"` / `"interleaved"`
    /// for audio. Defaults to `"hwc"` and `"planar"` respectively.
    pub layout: Option<String>,
}

/// Sample rate assumed by `payloadType: "audio"` when the caller omits one.
const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// Recognized image channel counts mapped to a color space.
fn infer_color_space(channels: u32) -> Option<ColorSpace> {
    match channels {
        1 => Some(ColorSpace::Grayscale),
        3 => Some(ColorSpace::Rgb),
        4 => Some(ColorSpace::Rgba),
        _ => None,
    }
}

fn parse_color_space(spec: &str) -> napi::Result<ColorSpace> {
    match spec.to_ascii_lowercase().as_str() {
        "gray" | "grey" | "grayscale" => Ok(ColorSpace::Grayscale),
        "rgb" => Ok(ColorSpace::Rgb),
        "rgba" => Ok(ColorSpace::Rgba),
        "bgr" => Ok(ColorSpace::Bgr),
        "bgra" => Ok(ColorSpace::Bgra),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
                "Unknown color space '{}' (expected one of: grayscale, rgb, rgba, bgr, bgra)",
                other
            ),
        )),
    }
}

fn parse_image_layout(spec: &str) -> napi::Result<ImageLayout> {
    match spec.to_ascii_lowercase().as_str() {
        "hwc" | "interleaved" => Ok(ImageLayout::Hwc),
        "chw" | "planar" => Ok(ImageLayout::Chw),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Unknown image layout '{}' (expected 'hwc' or 'chw')", other),
        )),
    }
}

fn parse_audio_layout(spec: &str) -> napi::Result<AudioLayout> {
    match spec.to_ascii_lowercase().as_str() {
        "planar" => Ok(AudioLayout::Planar),
        "interleaved" => Ok(AudioLayout::Interleaved),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
                "Unknown audio layout '{}' (expected 'planar' or 'interleaved')",
                other
            ),
        )),
    }
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
        Payload::Scalar(t) => Ok(tensor_to_morflow_tensor(&t)),
        Payload::Arg(_) => Err(napi::Error::new(
            napi::Status::GenericFailure,
            "Argument payload returned where a tensor was expected",
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
            if !bytes.len().is_multiple_of(4) {
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

    // An explicit payloadType chooses the payload type outright.
    if let Some(kind) = input.payload_type.as_deref() {
        match kind.to_ascii_lowercase().as_str() {
            "tensor" => return Ok(Payload::Tensor(tensor)),
            "image" => {
                let color_space = match input.color_space.as_deref() {
                    Some(spec) => parse_color_space(spec)?,
                    None => tensor
                        .shape
                        .get(2)
                        .copied()
                        .and_then(|c| infer_color_space(c as u32))
                        .ok_or_else(|| {
                            napi::Error::new(
                                napi::Status::InvalidArg,
                                format!(
                                    "payloadType \"image\" could not infer a color space from shape {:?}; \
                                     pass colorSpace: \"grayscale\", \"rgb\", \"rgba\", \"bgr\", or \"bgra\"",
                                    tensor.shape.as_slice()
                                ),
                            )
                        })?,
                };
                let layout = match input.layout.as_deref() {
                    Some(spec) => parse_image_layout(spec)?,
                    None => ImageLayout::Hwc,
                };
                let img = Image::new(tensor, color_space, layout).map_err(|e| {
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        format!("Invalid image payload: {}", e),
                    )
                })?;
                return Ok(Payload::Image(img));
            }
            "audio" => {
                let layout = match input.layout.as_deref() {
                    Some(spec) => parse_audio_layout(spec)?,
                    None => AudioLayout::Planar,
                };
                let channel_layout = match input.channels {
                    Some(ch) => AudioChannelLayout::from_channel_count(ch as usize),
                    None => AudioChannelLayout::from_channel_count(
                        tensor.shape.first().copied().unwrap_or(1),
                    ),
                };
                let aud = Audio::new(
                    tensor,
                    input.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE),
                    channel_layout,
                    layout,
                )
                .map_err(|e| {
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        format!("Invalid audio payload: {}", e),
                    )
                })?;
                return Ok(Payload::Audio(aud));
            }
            other => {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "Unknown payloadType '{}' (expected one of: tensor, image, audio)",
                        other
                    ),
                ))
            }
        }
    }

    // No payloadType given: a bare array is always a plain Tensor. There is no
    // type inference; set payloadType to "image" or "audio" to send something
    // else.
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

/// A heterogeneous host value bound to a pipeline parameter in declaration
/// order: a tensor/audio/image object, a typed array, a byte buffer, a plain
/// number (Scalar or `*Arg`), a string (`StrArg`/`Bytes`), or a boolean
/// (`BoolArg`).
type HostArg = Either6<TensorInput, Float32Array, Buffer, f64, String, bool>;

fn host_arg_to_payload(arg: HostArg) -> napi::Result<Payload> {
    match arg {
        Either6::A(tensor_input) => tensor_input_to_payload(tensor_input),
        Either6::B(f32_arr) => float32_array_to_payload(f32_arr),
        Either6::C(buf) => Ok(buffer_to_payload(buf)),
        Either6::D(f) => Ok(text_payload(&format!("{}", f))),
        Either6::E(s) => Ok(text_payload(&s)),
        Either6::F(b) => Ok(text_payload(&b.to_string())),
    }
}

fn text_payload(text: &str) -> Payload {
    Payload::Data {
        buffer: RVec::from(text.as_bytes().to_vec()),
    }
}

fn collect_host_args(args: Vec<HostArg>) -> napi::Result<Vec<Payload>> {
    args.into_iter().map(host_arg_to_payload).collect()
}

/// Libuv worker task for asynchronous single-output pipeline execution with
/// positional parameters.
pub struct AsyncRunArgsTask {
    pipeline: MorflowPipeline,
    payloads: Vec<Payload>,
}

impl Task for AsyncRunArgsTask {
    type Output = MorflowTensor;
    type JsValue = MorflowTensor;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let payloads = std::mem::take(&mut self.payloads);
        let outputs = self.pipeline.run_args(payloads).map_err(map_error)?;
        let single = outputs.into_single().map_err(map_error)?;
        payload_to_morflow_tensor(single)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

/// Libuv worker task for asynchronous multi-output pipeline execution with
/// positional parameters.
pub struct AsyncRunAllArgsTask {
    pipeline: MorflowPipeline,
    payloads: Vec<Payload>,
}

impl Task for AsyncRunAllArgsTask {
    type Output = Vec<(String, MorflowTensor)>;
    type JsValue = napi::JsObject;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let payloads = std::mem::take(&mut self.payloads);
        let outputs = self.pipeline.run_args(payloads).map_err(map_error)?;
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

    /// Executes the pipeline synchronously with positional parameters in
    /// declaration order. Parameters without a supplied value use their
    /// declared default.
    #[napi]
    pub fn run_sync_args(&mut self, args: Vec<HostArg>) -> napi::Result<MorflowTensor> {
        let payloads = collect_host_args(args)?;
        let outputs = self.inner.run_args(payloads).map_err(map_error)?;
        let single = outputs.into_single().map_err(map_error)?;
        payload_to_morflow_tensor(single)
    }

    /// Executes the pipeline synchronously returning a map of all named output
    /// streams, binding positional parameters in declaration order.
    #[napi]
    pub fn run_sync_all_args(
        &mut self,
        env: Env,
        args: Vec<HostArg>,
    ) -> napi::Result<napi::JsObject> {
        let payloads = collect_host_args(args)?;
        let outputs = self.inner.run_args(payloads).map_err(map_error)?;
        let mut obj = env.create_object()?;
        for (name, out_payload) in outputs.into_iter() {
            let tensor_obj = payload_to_morflow_tensor(out_payload)?;
            obj.set_named_property(&name, tensor_obj)?;
        }
        Ok(obj)
    }

    /// Executes the pipeline asynchronously on a worker thread with positional
    /// parameters in declaration order, returning a Promise<MorflowTensor>.
    #[napi]
    pub fn run_args(&self, args: Vec<HostArg>) -> napi::Result<AsyncTask<AsyncRunArgsTask>> {
        let payloads = collect_host_args(args)?;
        Ok(AsyncTask::new(AsyncRunArgsTask {
            pipeline: self.inner.clone(),
            payloads,
        }))
    }

    /// Executes the pipeline asynchronously on a worker thread with positional
    /// parameters, returning a Promise<Record<string, MorflowTensor>>.
    #[napi]
    pub fn run_all_args(&self, args: Vec<HostArg>) -> napi::Result<AsyncTask<AsyncRunAllArgsTask>> {
        let payloads = collect_host_args(args)?;
        Ok(AsyncTask::new(AsyncRunAllArgsTask {
            pipeline: self.inner.clone(),
            payloads,
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

/// Executes the Morflow CLI with the specified command-line arguments.
#[napi]
pub fn run_cli(args: Vec<String>) -> napi::Result<i32> {
    let mut full_args = vec!["morflow".to_string()];
    full_args.extend(args);
    Ok(morflow::cli::run_cli(full_args))
}
