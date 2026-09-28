use std::path::Path;

use abi_stable::std_types::RVec;
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pipeline::{Morflow, MorflowError, MorflowPipeline};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyString};

fn map_error(err: MorflowError) -> PyErr {
    match err {
        MorflowError::Io(e) => PyIOError::new_err(e.to_string()),
        MorflowError::Parse(e) => PyValueError::new_err(format!("Parse error: {}", e)),
        MorflowError::Compile(e) => PyValueError::new_err(format!("Compile error: {}", e)),
        MorflowError::Plugin(e) => PyRuntimeError::new_err(format!("Plugin error: {}", e)),
        MorflowError::Execution(e) => PyRuntimeError::new_err(format!("Execution error: {}", e)),
        MorflowError::TypeMismatch(e) => PyTypeError::new_err(format!("Type error: {}", e)),
    }
}

/// Convert a Python object (NumPy array, bytes, str, int, float) to a Morflow Payload.
fn py_any_to_payload(obj: &Bound<'_, PyAny>) -> PyResult<Payload> {
    // 1. Try NumPy Float32
    if let Ok(readonly) = obj.extract::<PyReadonlyArrayDyn<f32>>() {
        let shape: Vec<usize> = readonly.shape().to_vec();
        let slice = readonly
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Non-contiguous numpy array: {}", e)))?;
        let tensor = Tensor::from_f32_shape(slice, shape)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // If shape is 3D [H, W, C], interpret as Image for convenience
        if tensor.rank() == 3 {
            let channels = tensor.shape[2];
            let cs = match channels {
                1 => ColorSpace::Grayscale,
                3 => ColorSpace::Rgb,
                4 => ColorSpace::Rgba,
                _ => ColorSpace::Rgb,
            };
            if let Ok(img) = Image::new(tensor.clone(), cs, ImageLayout::Hwc) {
                return Ok(Payload::Image(img));
            }
        }
        // If shape is 2D [channels, samples], interpret as Audio
        if tensor.rank() == 2 && tensor.shape[0] <= 8 {
            let ch = tensor.shape[0];
            let layout = AudioChannelLayout::from_channel_count(ch);
            if let Ok(aud) = Audio::new(tensor.clone(), 44100, layout, AudioLayout::Planar) {
                return Ok(Payload::Audio(aud));
            }
        }

        return Ok(Payload::Tensor(tensor));
    }

    // 2. Try NumPy UInt8
    if let Ok(readonly) = obj.extract::<PyReadonlyArrayDyn<u8>>() {
        let shape: Vec<usize> = readonly.shape().to_vec();
        let slice = readonly
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Non-contiguous numpy array: {}", e)))?;
        let tensor = Tensor::from_rvec_u8(RVec::from(slice.to_vec()), shape, TensorDType::U8)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if tensor.rank() == 3 {
            let channels = tensor.shape[2];
            let cs = match channels {
                1 => ColorSpace::Grayscale,
                3 => ColorSpace::Rgb,
                4 => ColorSpace::Rgba,
                _ => ColorSpace::Rgb,
            };
            if let Ok(img) = Image::new(tensor.clone(), cs, ImageLayout::Hwc) {
                return Ok(Payload::Image(img));
            }
        }

        return Ok(Payload::Tensor(tensor));
    }

    // 3. Try NumPy Float64 (convert to f32 tensor)
    if let Ok(readonly) = obj.extract::<PyReadonlyArrayDyn<f64>>() {
        let shape: Vec<usize> = readonly.shape().to_vec();
        let slice = readonly
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Non-contiguous numpy array: {}", e)))?;
        let f32_vec: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&f32_vec, shape)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(Payload::Tensor(tensor));
    }

    // 4. Try NumPy Int32
    if let Ok(readonly) = obj.extract::<PyReadonlyArrayDyn<i32>>() {
        let shape: Vec<usize> = readonly.shape().to_vec();
        let slice = readonly
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Non-contiguous numpy array: {}", e)))?;
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * std::mem::size_of::<i32>(),
            )
        };
        let tensor = Tensor::from_rvec_u8(RVec::from(byte_slice.to_vec()), shape, TensorDType::I32)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(Payload::Tensor(tensor));
    }

    // 5. Try Raw Bytes / Bytearray
    if let Ok(bytes) = obj.extract::<&[u8]>() {
        return Ok(Payload::Data {
            buffer: RVec::from(bytes.to_vec()),
        });
    }

    // 6. Try String
    if let Ok(py_str) = obj.downcast::<PyString>() {
        let s = py_str.to_str()?;
        return Ok(Payload::Data {
            buffer: RVec::from(s.as_bytes().to_vec()),
        });
    }

    // 7. Try Integer
    if let Ok(py_int) = obj.downcast::<PyInt>() {
        let i: i64 = py_int.extract()?;
        return Ok(Payload::Data {
            buffer: RVec::from(i.to_string().into_bytes()),
        });
    }

    // 8. Try Float
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
    }
}

fn tensor_to_py<'py>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    let shape_ix = IxDyn(tensor.shape.as_slice());
    match tensor.dtype {
        TensorDType::F32 => {
            let data = tensor.to_vec_f32();
            let arr = ArrayD::from_shape_vec(shape_ix, data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = arr.into_pyarray(py);
            Ok(py_arr.into_any())
        }
        TensorDType::U8 => {
            let data = tensor.to_contiguous_bytes().into_vec();
            let arr = ArrayD::from_shape_vec(shape_ix, data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = arr.into_pyarray(py);
            Ok(py_arr.into_any())
        }
        TensorDType::I32 => {
            let i32_vec = if let Some(slice) = tensor.as_i32_slice() {
                slice.to_vec()
            } else {
                let bytes = tensor.to_contiguous_bytes();
                let i32_slice: &[i32] = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const i32,
                        bytes.len() / std::mem::size_of::<i32>(),
                    )
                };
                i32_slice.to_vec()
            };
            let arr = ArrayD::from_shape_vec(shape_ix, i32_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = arr.into_pyarray(py);
            Ok(py_arr.into_any())
        }
        _ => {
            // Default fallback as raw bytes
            let bytes = tensor.to_contiguous_bytes();
            Ok(PyBytes::new(py, bytes.as_slice()).into_any())
        }
    }
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

    /// Preloads and warms up all declared action plugins in memory.
    fn warmup(&self) -> PyResult<()> {
        self.inner.warmup().map_err(map_error)
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

/// Python module initialization.
#[pymodule]
fn _morflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_str, m)?)?;
    m.add_class::<PyPipeline>()?;
    Ok(())
}
