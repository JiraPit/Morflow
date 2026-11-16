//! C interface to the Morflow engine.
//!
//! This is the native layer behind the Dart package: Dart calls these
//! functions through `dart:ffi`, and application code never sees them. Input
//! conversion, output shapes and error messages match the Node.js binding, so
//! every host language behaves the same way.
//!
//! Conventions:
//! - Every function catches Rust panics and reports them as errors; a panic
//!   never unwinds into the caller.
//! - Strings are NUL-terminated UTF-8. Strings and results returned to the
//!   caller are freed with [`morflow_string_free`] and [`morflow_result_free`].
//! - Each run works on its own clone of the pipeline (actions are shared
//!   through reference counting), so a pipeline handle can be used from many
//!   threads at once.

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

use abi_stable::std_types::{RBox, RVec};
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use morflow::{Morflow, MorflowError, MorflowPipeline, PipelineOutputs};

/// A tensor, image, audio or Composite input, described like the Node.js
/// binding's `TensorInput`.
pub const MORFLOW_VALUE_TENSOR: u32 = 0;
/// Raw bytes (`Payload::Data`).
pub const MORFLOW_VALUE_BYTES: u32 = 1;
/// A number bound to a pipeline parameter.
pub const MORFLOW_VALUE_NUMBER: u32 = 2;
/// A string bound to a pipeline parameter.
pub const MORFLOW_VALUE_TEXT: u32 = 3;
/// A boolean bound to a pipeline parameter.
pub const MORFLOW_VALUE_BOOL: u32 = 4;

/// Sample rate assumed for `payload_type = "audio"` when none is given.
const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// A host value passed into a pipeline. Which fields are read depends on
/// `kind`; unused pointers may be null. Optional strings are null when absent,
/// and `sample_rate` / `channels` are negative when absent.
#[repr(C)]
pub struct MorflowValueIn {
    pub kind: u32,
    /// Element bytes (tensor), raw bytes (bytes) or UTF-8 text (text).
    pub data: *const u8,
    pub data_len: usize,
    pub shape: *const u64,
    pub rank: usize,
    pub dtype: *const c_char,
    pub payload_type: *const c_char,
    pub color_space: *const c_char,
    pub layout: *const c_char,
    pub sample_rate: i64,
    pub channels: i64,
    /// Components of a `payload_type = "composite"` tensor input.
    pub items: *const MorflowValueIn,
    pub item_count: usize,
    pub number: f64,
    pub flag: u8,
}

/// An emitted value. `dtype` is an element type such as `"f32"`, or
/// `"composite"` with the components in `items` and no data.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MorflowValueOut {
    pub dtype: *const c_char,
    pub shape: *const u64,
    pub rank: usize,
    pub data: *const u8,
    pub data_len: usize,
    pub items: *const MorflowValueOut,
    pub item_count: usize,
}

/// A compiled pipeline.
pub struct MorflowPipelineHandle {
    inner: MorflowPipeline,
}

/// The outcome of a run: an error, or one or more emitted values. A single-
/// output run has one unnamed value; an all-outputs run has one named value per
/// emitted stream, sorted by name.
pub struct MorflowResult {
    error: Option<CString>,
    names: Vec<Option<CString>>,
    values: Vec<OutNode>,
}

/// Called once from a worker thread when an asynchronous run finishes. The
/// callee owns `result` and frees it with [`morflow_result_free`].
pub type MorflowCallback = extern "C" fn(request_id: i64, result: *mut MorflowResult);

// ---------------------------------------------------------------------------
// Errors

fn map_error(err: MorflowError) -> String {
    match err {
        MorflowError::Io(e) => format!("IO error: {}", e),
        MorflowError::Parse(e) => format!("Parse error: {}", e),
        MorflowError::Compile(e) => format!("Compile error: {}", e),
        MorflowError::Action(e) => format!("Action error: {}", e),
        MorflowError::Execution(e) => format!("Execution error: {}", e),
        MorflowError::TypeMismatch(e) => format!("Type error: {}", e),
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    let detail = panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string());
    format!("Internal error: {}", detail)
}

/// Converts text into a C string, replacing interior NULs so error messages
/// are never lost.
fn c_string(text: String) -> CString {
    CString::new(text.replace('\0', "\u{FFFD}")).expect("NULs were replaced")
}

fn error_result(message: String) -> *mut MorflowResult {
    Box::into_raw(Box::new(MorflowResult {
        error: Some(c_string(message)),
        names: Vec::new(),
        values: Vec::new(),
    }))
}

unsafe fn set_error(error_out: *mut *mut c_char, message: String) {
    if !error_out.is_null() {
        *error_out = c_string(message).into_raw();
    }
}

// ---------------------------------------------------------------------------
// Input conversion (mirrors the Node.js binding's TensorInput handling)

unsafe fn opt_str<'a>(ptr: *const c_char, field: &str) -> Result<Option<&'a str>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map(Some)
        .map_err(|_| format!("{} is not valid UTF-8", field))
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

fn infer_color_space(channels: usize) -> Option<ColorSpace> {
    match channels {
        1 => Some(ColorSpace::Grayscale),
        3 => Some(ColorSpace::Rgb),
        4 => Some(ColorSpace::Rgba),
        _ => None,
    }
}

fn parse_color_space(spec: &str) -> Result<ColorSpace, String> {
    match spec.to_ascii_lowercase().as_str() {
        "gray" | "grey" | "grayscale" => Ok(ColorSpace::Grayscale),
        "rgb" => Ok(ColorSpace::Rgb),
        "rgba" => Ok(ColorSpace::Rgba),
        "bgr" => Ok(ColorSpace::Bgr),
        "bgra" => Ok(ColorSpace::Bgra),
        other => Err(format!(
            "Unknown color space '{}' (expected one of: grayscale, rgb, rgba, bgr, bgra)",
            other
        )),
    }
}

fn parse_image_layout(spec: &str) -> Result<ImageLayout, String> {
    match spec.to_ascii_lowercase().as_str() {
        "hwc" | "interleaved" => Ok(ImageLayout::Hwc),
        "chw" | "planar" => Ok(ImageLayout::Chw),
        other => Err(format!(
            "Unknown image layout '{}' (expected 'hwc' or 'chw')",
            other
        )),
    }
}

fn parse_audio_layout(spec: &str) -> Result<AudioLayout, String> {
    match spec.to_ascii_lowercase().as_str() {
        "planar" => Ok(AudioLayout::Planar),
        "interleaved" => Ok(AudioLayout::Interleaved),
        other => Err(format!(
            "Unknown audio layout '{}' (expected 'planar' or 'interleaved')",
            other
        )),
    }
}

fn text_payload(text: &str) -> Payload {
    Payload::Data {
        buffer: RVec::from(text.as_bytes().to_vec()),
    }
}

unsafe fn tensor_input_to_payload(value: &MorflowValueIn) -> Result<Payload, String> {
    let payload_type = opt_str(value.payload_type, "payloadType")?;

    // A Composite carries its components in `items` rather than in `data`.
    if payload_type.is_some_and(|kind| kind.eq_ignore_ascii_case("composite")) {
        let payloads = slice(value.items, value.item_count)
            .iter()
            .map(|item| tensor_input_to_payload(item))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Payload::Composite(RVec::from(payloads)));
    }

    let shape: Vec<usize> = slice(value.shape, value.rank)
        .iter()
        .map(|&d| d as usize)
        .collect();
    let dtype_name = opt_str(value.dtype, "dtype")?.unwrap_or("f32");
    let dtype = TensorDType::from_name(dtype_name).ok_or_else(|| {
        format!(
            "Unknown dtype '{}' (expected one of: f32, f64, u8, i8, i16, i32, i64, u32, u64)",
            dtype_name
        )
    })?;
    let bytes = slice(value.data, value.data_len);
    if !bytes.len().is_multiple_of(dtype.element_size()) {
        return Err(format!(
            "Data buffer size is not a multiple of {} for a {} tensor",
            dtype.element_size(),
            dtype.name()
        ));
    }
    let tensor = Tensor::from_rvec_u8(RVec::from(bytes.to_vec()), shape, dtype)
        .map_err(|e| e.to_string())?;

    // An explicit payloadType chooses the payload type outright.
    if let Some(kind) = payload_type {
        match kind.to_ascii_lowercase().as_str() {
            "tensor" => return Ok(Payload::Tensor(tensor)),
            "image" => {
                let color_space = match opt_str(value.color_space, "colorSpace")? {
                    Some(spec) => parse_color_space(spec)?,
                    None => tensor
                        .shape
                        .get(2)
                        .copied()
                        .and_then(infer_color_space)
                        .ok_or_else(|| {
                            format!(
                                "payloadType \"image\" could not infer a color space from shape {:?}; \
                                 pass colorSpace: \"grayscale\", \"rgb\", \"rgba\", \"bgr\", or \"bgra\"",
                                tensor.shape.as_slice()
                            )
                        })?,
                };
                let layout = match opt_str(value.layout, "layout")? {
                    Some(spec) => parse_image_layout(spec)?,
                    None => ImageLayout::Hwc,
                };
                let img = Image::new(tensor, color_space, layout)
                    .map_err(|e| format!("Invalid image payload: {}", e))?;
                return Ok(Payload::Image(img));
            }
            "audio" => {
                let layout = match opt_str(value.layout, "layout")? {
                    Some(spec) => parse_audio_layout(spec)?,
                    None => AudioLayout::Planar,
                };
                let channels = if value.channels >= 0 {
                    value.channels as usize
                } else {
                    tensor.shape.first().copied().unwrap_or(1)
                };
                let sample_rate = if value.sample_rate >= 0 {
                    u32::try_from(value.sample_rate)
                        .map_err(|_| format!("sampleRate {} is too large", value.sample_rate))?
                } else {
                    DEFAULT_SAMPLE_RATE
                };
                let aud = Audio::new(
                    tensor,
                    sample_rate,
                    AudioChannelLayout::from_channel_count(channels),
                    layout,
                )
                .map_err(|e| format!("Invalid audio payload: {}", e))?;
                return Ok(Payload::Audio(aud));
            }
            other => {
                return Err(format!(
                    "Unknown payloadType '{}' (expected tensor, image, audio, or composite)",
                    other
                ))
            }
        }
    }

    // No payloadType given: a bare array is always a plain Tensor.
    Ok(Payload::Tensor(tensor))
}

unsafe fn value_to_payload(value: &MorflowValueIn) -> Result<Payload, String> {
    match value.kind {
        MORFLOW_VALUE_TENSOR => tensor_input_to_payload(value),
        MORFLOW_VALUE_BYTES => Ok(Payload::Data {
            buffer: RVec::from(slice(value.data, value.data_len).to_vec()),
        }),
        MORFLOW_VALUE_NUMBER => Ok(text_payload(&format!("{}", value.number))),
        MORFLOW_VALUE_TEXT => {
            let text = std::str::from_utf8(slice(value.data, value.data_len))
                .map_err(|_| "String argument is not valid UTF-8".to_string())?;
            Ok(text_payload(text))
        }
        MORFLOW_VALUE_BOOL => Ok(text_payload(if value.flag != 0 { "true" } else { "false" })),
        other => Err(format!("Unknown value kind {}", other)),
    }
}

/// The payload for a single-input run: absent input is an empty byte buffer.
unsafe fn input_payload(input: *const MorflowValueIn) -> Result<Payload, String> {
    if input.is_null() {
        Ok(Payload::Data {
            buffer: RVec::new(),
        })
    } else {
        value_to_payload(&*input)
    }
}

unsafe fn args_payloads(args: *const MorflowValueIn, count: usize) -> Result<Vec<Payload>, String> {
    slice(args, count)
        .iter()
        .map(|v| value_to_payload(v))
        .collect()
}

// ---------------------------------------------------------------------------
// Output conversion (mirrors the Node.js binding's MorflowTensor)

/// Owned storage behind a [`MorflowValueOut`]. Every buffer lives on the heap,
/// so the raw pointers in `out` stay valid when the node itself moves.
struct OutNode {
    _dtype: CString,
    _shape: Box<[u64]>,
    _data: Box<[u8]>,
    _children: Box<[OutNode]>,
    _children_out: Box<[MorflowValueOut]>,
    out: MorflowValueOut,
}

impl OutNode {
    fn new(dtype: &str, shape: Vec<u64>, data: Vec<u8>, children: Vec<OutNode>) -> OutNode {
        let dtype = c_string(dtype.to_string());
        let shape = shape.into_boxed_slice();
        let data = data.into_boxed_slice();
        let children = children.into_boxed_slice();
        let children_out: Box<[MorflowValueOut]> = children.iter().map(|c| c.out).collect();
        let out = MorflowValueOut {
            dtype: dtype.as_ptr(),
            shape: shape.as_ptr(),
            rank: shape.len(),
            data: data.as_ptr(),
            data_len: data.len(),
            items: children_out.as_ptr(),
            item_count: children_out.len(),
        };
        OutNode {
            _dtype: dtype,
            _shape: shape,
            _data: data,
            _children: children,
            _children_out: children_out,
            out,
        }
    }

    fn tensor(tensor: &Tensor) -> OutNode {
        let data = match tensor.as_bytes() {
            Some(bytes) => bytes.to_vec(),
            None => tensor.to_contiguous_bytes().to_vec(),
        };
        let shape = tensor.shape.iter().map(|&d| d as u64).collect();
        OutNode::new(tensor.dtype.name(), shape, data, Vec::new())
    }
}

fn payload_to_node(payload: Payload) -> Result<OutNode, String> {
    match payload {
        Payload::Tensor(t) | Payload::Scalar(t) => Ok(OutNode::tensor(&t)),
        Payload::Image(img) => Ok(OutNode::tensor(&img.tensor)),
        Payload::Audio(aud) => Ok(OutNode::tensor(&aud.tensor)),
        Payload::Data { buffer } => Ok(OutNode::new(
            "u8",
            vec![buffer.len() as u64],
            buffer.to_vec(),
            Vec::new(),
        )),
        Payload::WithArgs { payload, .. } => payload_to_node(RBox::into_inner(payload)),
        Payload::Error(err) => Err(err.to_string()),
        Payload::Composite(items) => {
            let count = items.len() as u64;
            let children = items
                .into_iter()
                .map(payload_to_node)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(OutNode::new("composite", vec![count], Vec::new(), children))
        }
        Payload::Arg(_) => Err("Argument payload returned where a tensor was expected".to_string()),
    }
}

fn outputs_to_result(outputs: PipelineOutputs, all: bool) -> Result<MorflowResult, String> {
    if !all {
        let single = outputs.into_single().map_err(map_error)?;
        return Ok(MorflowResult {
            error: None,
            names: vec![None],
            values: vec![payload_to_node(single)?],
        });
    }
    let mut entries: Vec<(String, Payload)> = outputs.into_map().into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut names = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (name, payload) in entries {
        values.push(payload_to_node(payload)?);
        names.push(Some(c_string(name)));
    }
    Ok(MorflowResult {
        error: None,
        names,
        values,
    })
}

// ---------------------------------------------------------------------------
// Execution

enum Call {
    Input(Payload),
    Args(Vec<Payload>),
}

fn execute(mut pipeline: MorflowPipeline, call: Call, all: bool) -> *mut MorflowResult {
    let outcome = catch_unwind(AssertUnwindSafe(move || {
        let outputs = match call {
            Call::Input(payload) => pipeline.run(payload),
            Call::Args(payloads) => pipeline.run_args(payloads),
        }
        .map_err(map_error)?;
        outputs_to_result(outputs, all)
    }));
    match outcome {
        Ok(Ok(result)) => Box::into_raw(Box::new(result)),
        Ok(Err(message)) => error_result(message),
        Err(panic) => error_result(panic_message(panic)),
    }
}

/// Runs `call` on a worker thread and delivers the result through `callback`.
/// Returns null when the run started, or an error result when it could not.
fn spawn(
    pipeline: MorflowPipeline,
    call: Call,
    all: bool,
    callback: MorflowCallback,
    request_id: i64,
) -> *mut MorflowResult {
    let started = std::thread::Builder::new()
        .name("morflow-run".to_string())
        .spawn(move || {
            let result = execute(pipeline, call, all);
            callback(request_id, result);
        });
    match started {
        Ok(_) => ptr::null_mut(),
        Err(e) => error_result(format!("Could not start a worker thread: {}", e)),
    }
}

/// Converts inputs on the calling thread, so the caller may free them as soon
/// as the function returns, then runs synchronously or starts an async run.
unsafe fn dispatch(
    pipeline: *const MorflowPipelineHandle,
    convert: impl FnOnce() -> Result<Call, String>,
    all: u8,
    callback: Option<(MorflowCallback, i64)>,
) -> *mut MorflowResult {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        if pipeline.is_null() {
            return Err("Pipeline handle is null".to_string());
        }
        let pipeline = &*pipeline;
        Ok((pipeline.inner.clone(), convert()?))
    }));
    let (pipeline, call) = match outcome {
        Ok(Ok(parts)) => parts,
        Ok(Err(message)) => return error_result(message),
        Err(panic) => return error_result(panic_message(panic)),
    };
    match callback {
        None => execute(pipeline, call, all != 0),
        Some((callback, request_id)) => spawn(pipeline, call, all != 0, callback, request_id),
    }
}

// ---------------------------------------------------------------------------
// Exported functions

/// The version of this library, matching the engine version.
#[no_mangle]
pub extern "C" fn morflow_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

unsafe fn compile(
    error_out: *mut *mut c_char,
    build: impl FnOnce() -> Result<MorflowPipeline, String>,
) -> *mut MorflowPipelineHandle {
    match catch_unwind(AssertUnwindSafe(build)) {
        Ok(Ok(inner)) => Box::into_raw(Box::new(MorflowPipelineHandle { inner })),
        Ok(Err(message)) => {
            set_error(error_out, message);
            ptr::null_mut()
        }
        Err(panic) => {
            set_error(error_out, panic_message(panic));
            ptr::null_mut()
        }
    }
}

/// Compiles and loads a `.morf` pipeline file. Returns null on failure and
/// stores the message in `*error_out`.
///
/// # Safety
/// `path` must be a valid NUL-terminated string; `error_out` must be null or
/// writable.
#[no_mangle]
pub unsafe extern "C" fn morflow_load(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut MorflowPipelineHandle {
    compile(error_out, || {
        let path = opt_str(path, "path")?.ok_or("Pipeline path is null")?;
        Morflow::load(Path::new(path)).map_err(map_error)
    })
}

/// Compiles `.morf` source text. Returns null on failure and stores the
/// message in `*error_out`.
///
/// # Safety
/// `source` must be a valid NUL-terminated string; `error_out` must be null or
/// writable.
#[no_mangle]
pub unsafe extern "C" fn morflow_from_str(
    source: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut MorflowPipelineHandle {
    compile(error_out, || {
        let source = opt_str(source, "source")?.ok_or("Pipeline source is null")?;
        Morflow::from_str(source).map_err(map_error)
    })
}

/// Releases a pipeline. Runs already in progress keep their own clone.
///
/// # Safety
/// `pipeline` must be null or a handle from this library, freed only once.
#[no_mangle]
pub unsafe extern "C" fn morflow_pipeline_free(pipeline: *mut MorflowPipelineHandle) {
    if !pipeline.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(pipeline))));
    }
}

/// The number of parameters declared with `accept`.
///
/// # Safety
/// `pipeline` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn morflow_param_count(pipeline: *const MorflowPipelineHandle) -> usize {
    if pipeline.is_null() {
        return 0;
    }
    let pipeline = &*pipeline;
    pipeline.inner.params().len()
}

/// The name of parameter `index`, in declaration order, or null when out of
/// range. Free it with [`morflow_string_free`].
///
/// # Safety
/// `pipeline` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn morflow_param_name(
    pipeline: *const MorflowPipelineHandle,
    index: usize,
) -> *mut c_char {
    if pipeline.is_null() {
        return ptr::null_mut();
    }
    let pipeline = &*pipeline;
    match pipeline.inner.params().get(index) {
        Some(param) => c_string(param.name.clone()).into_raw(),
        None => ptr::null_mut(),
    }
}

/// Frees a string returned by this library.
///
/// # Safety
/// `text` must be null or a string from this library, freed only once.
#[no_mangle]
pub unsafe extern "C" fn morflow_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}

/// Runs the pipeline with one input (null for none). With `all` nonzero the
/// result holds every named output; otherwise exactly one output is expected.
///
/// # Safety
/// `pipeline` must be a live handle and `input` null or a valid value.
#[no_mangle]
pub unsafe extern "C" fn morflow_run(
    pipeline: *const MorflowPipelineHandle,
    input: *const MorflowValueIn,
    all: u8,
) -> *mut MorflowResult {
    dispatch(
        pipeline,
        || input_payload(input).map(Call::Input),
        all,
        None,
    )
}

/// Runs the pipeline with positional parameters in declaration order.
///
/// # Safety
/// `pipeline` must be a live handle and `args` point to `count` valid values.
#[no_mangle]
pub unsafe extern "C" fn morflow_run_args(
    pipeline: *const MorflowPipelineHandle,
    args: *const MorflowValueIn,
    count: usize,
    all: u8,
) -> *mut MorflowResult {
    dispatch(
        pipeline,
        || args_payloads(args, count).map(Call::Args),
        all,
        None,
    )
}

/// Like [`morflow_run`], on a worker thread. Inputs are converted before this
/// returns. Returns null when the run started (`callback` then fires exactly
/// once), or an error result when it did not (`callback` never fires).
///
/// # Safety
/// As for [`morflow_run`]; `callback` must stay callable until it fires.
#[no_mangle]
pub unsafe extern "C" fn morflow_run_async(
    pipeline: *const MorflowPipelineHandle,
    input: *const MorflowValueIn,
    all: u8,
    callback: MorflowCallback,
    request_id: i64,
) -> *mut MorflowResult {
    dispatch(
        pipeline,
        || input_payload(input).map(Call::Input),
        all,
        Some((callback, request_id)),
    )
}

/// Like [`morflow_run_args`], on a worker thread. See [`morflow_run_async`].
///
/// # Safety
/// As for [`morflow_run_args`]; `callback` must stay callable until it fires.
#[no_mangle]
pub unsafe extern "C" fn morflow_run_args_async(
    pipeline: *const MorflowPipelineHandle,
    args: *const MorflowValueIn,
    count: usize,
    all: u8,
    callback: MorflowCallback,
    request_id: i64,
) -> *mut MorflowResult {
    dispatch(
        pipeline,
        || args_payloads(args, count).map(Call::Args),
        all,
        Some((callback, request_id)),
    )
}

/// The error message, or null when the run succeeded. Owned by the result.
///
/// # Safety
/// `result` must be a live result.
#[no_mangle]
pub unsafe extern "C" fn morflow_result_error(result: *const MorflowResult) -> *const c_char {
    let result = &*result;
    match result.error.as_ref() {
        Some(message) => message.as_ptr(),
        None => ptr::null(),
    }
}

/// The number of emitted values.
///
/// # Safety
/// `result` must be a live result.
#[no_mangle]
pub unsafe extern "C" fn morflow_result_count(result: *const MorflowResult) -> usize {
    let result = &*result;
    result.values.len()
}

/// The output name of value `index`, or null for a single-output run or an
/// index out of range. Owned by the result.
///
/// # Safety
/// `result` must be a live result.
#[no_mangle]
pub unsafe extern "C" fn morflow_result_name(
    result: *const MorflowResult,
    index: usize,
) -> *const c_char {
    let result = &*result;
    match result.names.get(index) {
        Some(Some(name)) => name.as_ptr(),
        _ => ptr::null(),
    }
}

/// Value `index`, or null when out of range. Owned by the result.
///
/// # Safety
/// `result` must be a live result.
#[no_mangle]
pub unsafe extern "C" fn morflow_result_value(
    result: *const MorflowResult,
    index: usize,
) -> *const MorflowValueOut {
    let result = &*result;
    match result.values.get(index) {
        Some(node) => &node.out,
        None => ptr::null(),
    }
}

/// Frees a result and every value it owns.
///
/// # Safety
/// `result` must be null or a result from this library, freed only once.
#[no_mangle]
pub unsafe extern "C" fn morflow_result_free(result: *mut MorflowResult) {
    if !result.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(result))));
    }
}

#[cfg(test)]
mod tests;
