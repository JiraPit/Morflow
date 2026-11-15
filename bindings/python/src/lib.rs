use std::path::Path;

use abi_stable::std_types::RVec;
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use morflow::{Morflow, MorflowError, MorflowPipeline};
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

fn map_error(err: MorflowError) -> PyErr {
    match err {
        MorflowError::Io(e) => PyIOError::new_err(e.to_string()),
        MorflowError::Parse(e) => PyValueError::new_err(format!("Parse error: {}", e)),
        MorflowError::Compile(e) => PyValueError::new_err(format!("Compile error: {}", e)),
        MorflowError::Action(e) => PyRuntimeError::new_err(format!("Action error: {}", e)),
        MorflowError::Execution(e) => PyRuntimeError::new_err(format!("Execution error: {}", e)),
        MorflowError::TypeMismatch(e) => PyTypeError::new_err(format!("Type error: {}", e)),
    }
}

/// Sample rate assumed by `morflow.Audio` when the caller does not provide one.
const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// Recognized image channel counts mapped to a color space.
fn infer_color_space(channels: usize) -> Option<ColorSpace> {
    match channels {
        1 => Some(ColorSpace::Grayscale),
        3 => Some(ColorSpace::Rgb),
        4 => Some(ColorSpace::Rgba),
        _ => None,
    }
}

fn parse_color_space(spec: &str) -> PyResult<ColorSpace> {
    match spec.to_ascii_lowercase().as_str() {
        "gray" | "grey" | "grayscale" => Ok(ColorSpace::Grayscale),
        "rgb" => Ok(ColorSpace::Rgb),
        "rgba" => Ok(ColorSpace::Rgba),
        "bgr" => Ok(ColorSpace::Bgr),
        "bgra" => Ok(ColorSpace::Bgra),
        other => Err(PyValueError::new_err(format!(
            "Unknown color space '{}' (expected one of: grayscale, rgb, rgba, bgr, bgra)",
            other
        ))),
    }
}

fn parse_image_layout(spec: &str) -> PyResult<ImageLayout> {
    match spec.to_ascii_lowercase().as_str() {
        "hwc" | "interleaved" => Ok(ImageLayout::Hwc),
        "chw" | "planar" => Ok(ImageLayout::Chw),
        other => Err(PyValueError::new_err(format!(
            "Unknown image layout '{}' (expected 'hwc' or 'chw')",
            other
        ))),
    }
}

fn parse_audio_layout(spec: &str) -> PyResult<AudioLayout> {
    match spec.to_ascii_lowercase().as_str() {
        "planar" => Ok(AudioLayout::Planar),
        "interleaved" => Ok(AudioLayout::Interleaved),
        other => Err(PyValueError::new_err(format!(
            "Unknown audio layout '{}' (expected 'planar' or 'interleaved')",
            other
        ))),
    }
}

/// Copies a NumPy array whose dtype is exactly `T` into a tensor of the
/// matching dtype. Returns `None` when `obj` is not such an array.
fn typed_ndarray_to_tensor<T>(obj: &Bound<'_, PyAny>) -> Option<PyResult<Tensor>>
where
    T: numpy::Element + core_types::numeric::Element,
{
    let readonly = obj.extract::<PyReadonlyArrayDyn<T>>().ok()?;
    let shape: Vec<usize> = readonly.shape().to_vec();
    Some((|| {
        let slice = readonly
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Non-contiguous numpy array: {}", e)))?;
        Tensor::from_vec_of(slice.to_vec(), shape).map_err(|e| PyValueError::new_err(e.to_string()))
    })())
}

/// Extracts a NumPy array of any tensor dtype (float32/64, int8/16/32/64,
/// uint8/32/64) into a plain tensor of the same dtype. Returns `None` when
/// `obj` is not a recognized array.
fn ndarray_to_tensor(obj: &Bound<'_, PyAny>) -> Option<PyResult<Tensor>> {
    typed_ndarray_to_tensor::<f32>(obj)
        .or_else(|| typed_ndarray_to_tensor::<f64>(obj))
        .or_else(|| typed_ndarray_to_tensor::<u8>(obj))
        .or_else(|| typed_ndarray_to_tensor::<i8>(obj))
        .or_else(|| typed_ndarray_to_tensor::<i16>(obj))
        .or_else(|| typed_ndarray_to_tensor::<i32>(obj))
        .or_else(|| typed_ndarray_to_tensor::<i64>(obj))
        .or_else(|| typed_ndarray_to_tensor::<u32>(obj))
        .or_else(|| typed_ndarray_to_tensor::<u64>(obj))
}

/// Images and audio hold `F32` (or `U8`) samples, so a float64 array sent as
/// one is converted to `F32`.
fn f64_samples_to_f32(tensor: Tensor) -> PyResult<Tensor> {
    if tensor.dtype != TensorDType::F64 {
        return Ok(tensor);
    }
    Tensor::from_vec_of(tensor.to_vec_as::<f32>(), tensor.shape.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Extracts the tensor from a supported NumPy array, or reports a clear error
/// for the explicit payload wrapper constructors.
fn require_ndarray_tensor(obj: &Bound<'_, PyAny>, wrapper: &str) -> PyResult<Tensor> {
    match ndarray_to_tensor(obj) {
        Some(Ok(tensor)) => Ok(tensor),
        Some(Err(e)) => Err(e),
        None => Err(PyTypeError::new_err(format!(
            "morflow.{}() expects a contiguous NumPy array, got {}",
            wrapper,
            obj.get_type().name()?
        ))),
    }
}

/// Forces a NumPy array to be passed as a plain `Payload::Tensor`.
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    data: Py<PyAny>,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Py<PyAny>) -> Self {
        Self { data }
    }
}

/// Forces a NumPy array to be passed as a `Payload::Image`.
#[pyclass(name = "Image")]
pub struct PyImage {
    data: Py<PyAny>,
    color: Option<String>,
    layout: Option<String>,
}

#[pymethods]
impl PyImage {
    #[new]
    #[pyo3(signature = (data, color=None, layout=None))]
    fn new(data: Py<PyAny>, color: Option<String>, layout: Option<String>) -> Self {
        Self {
            data,
            color,
            layout,
        }
    }
}

/// Forces a NumPy array to be passed as a `Payload::Audio`.
#[pyclass(name = "Audio")]
pub struct PyAudio {
    data: Py<PyAny>,
    sample_rate: u32,
    channels: Option<u32>,
    layout: Option<String>,
}

#[pymethods]
impl PyAudio {
    #[new]
    #[pyo3(signature = (data, sample_rate=DEFAULT_SAMPLE_RATE, channels=None, layout=None))]
    fn new(
        data: Py<PyAny>,
        sample_rate: u32,
        channels: Option<u32>,
        layout: Option<String>,
    ) -> Self {
        Self {
            data,
            sample_rate,
            channels,
            layout,
        }
    }
}

/// Convert a Python object (NumPy array, explicit payload wrapper, list or tuple,
/// bytes, str, int, float) to a Morflow Payload.
fn py_any_to_payload(obj: &Bound<'_, PyAny>) -> PyResult<Payload> {
    // 1. Explicit payload wrappers, which take precedence over auto-detection.
    if let Ok(forced) = obj.downcast::<PyTensor>() {
        let forced = forced.borrow();
        return Ok(Payload::Tensor(require_ndarray_tensor(
            forced.data.bind(obj.py()),
            "Tensor",
        )?));
    }

    if let Ok(forced) = obj.downcast::<PyImage>() {
        let forced = forced.borrow();
        let tensor =
            f64_samples_to_f32(require_ndarray_tensor(forced.data.bind(obj.py()), "Image")?)?;
        let color_space = match forced.color.as_deref() {
            Some(spec) => parse_color_space(spec)?,
            None => tensor
                .shape
                .get(2)
                .copied()
                .and_then(infer_color_space)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "morflow.Image() could not infer a color space from shape {:?}; \
                         pass color=\"grayscale\", \"rgb\", \"rgba\", \"bgr\", or \"bgra\"",
                        tensor.shape.as_slice()
                    ))
                })?,
        };
        let layout = match forced.layout.as_deref() {
            Some(spec) => parse_image_layout(spec)?,
            None => ImageLayout::Hwc,
        };
        let img = Image::new(tensor, color_space, layout)
            .map_err(|e| PyValueError::new_err(format!("Invalid image payload: {}", e)))?;
        return Ok(Payload::Image(img));
    }

    if let Ok(forced) = obj.downcast::<PyAudio>() {
        let forced = forced.borrow();
        let tensor =
            f64_samples_to_f32(require_ndarray_tensor(forced.data.bind(obj.py()), "Audio")?)?;
        let layout = match forced.layout.as_deref() {
            Some(spec) => parse_audio_layout(spec)?,
            None => AudioLayout::Planar,
        };
        let channel_layout = match forced.channels {
            Some(ch) => AudioChannelLayout::from_channel_count(ch as usize),
            None => {
                AudioChannelLayout::from_channel_count(tensor.shape.first().copied().unwrap_or(1))
            }
        };
        let aud = Audio::new(tensor, forced.sample_rate, channel_layout, layout)
            .map_err(|e| PyValueError::new_err(format!("Invalid audio payload: {}", e)))?;
        return Ok(Payload::Audio(aud));
    }

    // 2. Bare NumPy arrays are always a plain Tensor. There is no type
    //    inference: wrap the array in morflow.Image or morflow.Audio to send it
    //    as something other than a Tensor.
    if let Some(result) = ndarray_to_tensor(obj) {
        return Ok(Payload::Tensor(result?));
    }

    // 3. A list or tuple is a Composite with one component per item, so
    //    `[a, b]` feeds the two-tensor form of add/sub/mul/div/pow.
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items = obj
            .try_iter()?
            .map(|item| py_any_to_payload(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Payload::Composite(RVec::from(items)));
    }

    // 4. Bytes / Bytearray
    if let Ok(bytes) = obj.extract::<&[u8]>() {
        return Ok(Payload::Data {
            buffer: RVec::from(bytes.to_vec()),
        });
    }

    // 5. String
    if let Ok(py_str) = obj.downcast::<PyString>() {
        let s = py_str.to_cow()?;
        return Ok(Payload::Data {
            buffer: RVec::from(s.as_bytes().to_vec()),
        });
    }

    // 6. Integer
    if let Ok(py_int) = obj.downcast::<PyInt>() {
        let i: i64 = py_int.extract()?;
        return Ok(Payload::Data {
            buffer: RVec::from(i.to_string().into_bytes()),
        });
    }

    // 7. Float
    if let Ok(py_float) = obj.downcast::<PyFloat>() {
        let f: f64 = py_float.extract()?;
        return Ok(Payload::Data {
            buffer: RVec::from(f.to_string().into_bytes()),
        });
    }

    Err(PyTypeError::new_err(format!(
        "Unsupported payload type for Morflow: {}",
        obj.get_type().name()?
    )))
}

/// Convert a Morflow Payload into a Python object (NumPy array, bytes, or string).
fn payload_to_py<'py>(py: Python<'py>, payload: &Payload) -> PyResult<Bound<'py, PyAny>> {
    match payload {
        Payload::Tensor(tensor) => tensor_to_py(py, tensor),
        Payload::Image(img) => tensor_to_py(py, &img.tensor),
        Payload::Audio(aud) => tensor_to_py(py, &aud.tensor),
        Payload::Data { buffer } => {
            let py_bytes = PyBytes::new(py, buffer.as_slice());
            Ok(py_bytes.into_any())
        }
        Payload::WithArgs { payload, .. } => payload_to_py(py, payload),
        Payload::Error(err) => Err(PyRuntimeError::new_err(err.to_string())),
        Payload::Composite(items) => {
            let list = pyo3::types::PyList::empty(py);
            for item in items {
                list.append(payload_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        Payload::Scalar(tensor) => tensor_to_py(py, tensor),
        Payload::Arg(bytes) => {
            let py_str = PyString::new(py, &String::from_utf8_lossy(bytes.as_slice()));
            Ok(py_str.into_any())
        }
    }
}

/// Converts a tensor of any dtype into a NumPy array of the same dtype.
fn tensor_to_py<'py>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    fn typed<'py, T>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyAny>>
    where
        T: numpy::Element + core_types::numeric::Element,
    {
        let data = tensor.to_vec_of::<T>().unwrap_or_default();
        let arr = ArrayD::from_shape_vec(IxDyn(tensor.shape.as_slice()), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py).into_any())
    }
    core_types::match_dtype!(tensor.dtype,
        int I => typed::<I>(py, tensor),
        float F => typed::<F>(py, tensor),
    )
}

/// An executable Morflow pipeline instance.
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    inner: MorflowPipeline,
}

#[pymethods]
impl PyPipeline {
    /// Parameter names declared with `accept $param`.
    #[getter]
    fn params(&self) -> Vec<String> {
        self.inner.params().iter().map(|p| p.name.clone()).collect()
    }

    /// Executes the pipeline with a single input or positional arguments.
    /// Returns either a single NumPy array/object, or a dictionary of named outputs.
    #[pyo3(signature = (*args, **kwargs))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        args: &Bound<'py, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut positional_payloads = Vec::new();

        for item in args.iter() {
            positional_payloads.push(py_any_to_payload(&item)?);
        }

        // If kwargs are passed, bind them if positional is empty
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let _key: String = k.extract()?;
                positional_payloads.push(py_any_to_payload(&v)?);
            }
        }

        let outputs = py
            .allow_threads(|| {
                if positional_payloads.is_empty() {
                    self.inner.run_args(vec![])
                } else {
                    self.inner.run_args(positional_payloads)
                }
            })
            .map_err(map_error)?;

        // If pipeline has exactly 1 output (and unnamed or single emit), return the raw object directly
        if outputs.len() == 1 {
            let single = outputs.into_single().map_err(map_error)?;
            return payload_to_py(py, &single);
        }

        // For multi-output pipelines, return a Python dict mapping name -> output
        let dict = PyDict::new(py);
        for (name, payload) in outputs.iter() {
            let py_val = payload_to_py(py, payload)?;
            dict.set_item(name, py_val)?;
        }

        Ok(dict.into_any())
    }

    fn __repr__(&self) -> String {
        format!("<Morflow Pipeline params={:?}>", self.params())
    }
}

/// Compiles and loads a `.morf` pipeline file from disk.
#[pyfunction]
pub fn load(path: &str) -> PyResult<PyPipeline> {
    let p = Path::new(path);
    let pipeline = Morflow::load(p).map_err(map_error)?;
    Ok(PyPipeline { inner: pipeline })
}

/// Compiles a `.morf` pipeline DSL source string directly.
#[pyfunction]
pub fn from_str(source: &str) -> PyResult<PyPipeline> {
    let pipeline = Morflow::from_str(source).map_err(map_error)?;
    Ok(PyPipeline { inner: pipeline })
}

/// Executes the Morflow CLI with the specified command-line arguments.
#[pyfunction]
pub fn run_cli(args: Vec<String>) -> PyResult<i32> {
    let mut full_args = vec!["morflow".to_string()];
    full_args.extend(args);
    Ok(morflow::cli::run_cli(full_args))
}

/// Python module initialization.
#[pymodule]
fn _morflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_str, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyImage>()?;
    m.add_class::<PyAudio>()?;
    Ok(())
}
