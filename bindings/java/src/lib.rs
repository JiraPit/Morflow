use std::path::Path;

use abi_stable::std_types::RVec;
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use jni::objects::{
    JByteArray, JByteBuffer, JClass, JIntArray, JObject, JObjectArray, JString, JValue,
};
use jni::sys::{jint, jlong, jobject, jobjectArray};
use jni::JNIEnv;
use morflow::{Morflow, MorflowError, MorflowPipeline};

/// Sample rate assumed by `payloadType="audio"` when the caller omits one.
const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// The explicit payload type as passed across the JNI boundary by
/// `nativeRunDirect`. Each optional field is a plain string so the FFI signature
/// stays primitive; an empty string means "not specified".
#[repr(C)]
pub struct PayloadSpecJava {
    payload_type: JString<'static>,
    color_space: JString<'static>,
    layout: JString<'static>,
    sample_rate: jint,
    channels: jint,
}

impl PayloadSpecJava {
    fn opt_string(env: &mut JNIEnv, s: &JString) -> Result<Option<String>, String> {
        if s.is_null() {
            return Ok(None);
        }
        let v: String = env
            .get_string(s)
            .map(|v| v.into())
            .map_err(|e| format!("Failed to read payload spec string: {}", e))?;
        if v.is_empty() {
            Ok(None)
        } else {
            Ok(Some(v))
        }
    }

    fn into_rust(self, env: &mut JNIEnv) -> Result<PayloadSpec, String> {
        let kind = match Self::opt_string(env, &self.payload_type)? {
            Some(spec) => PayloadKind::parse(&spec)?,
            None => PayloadKind::Tensor,
        };
        Ok(PayloadSpec {
            kind,
            color_space: Self::opt_string(env, &self.color_space)?,
            sample_rate: if self.sample_rate > 0 {
                Some(self.sample_rate as u32)
            } else {
                None
            },
            channels: if self.channels > 0 {
                Some(self.channels as u32)
            } else {
                None
            },
            layout: Self::opt_string(env, &self.layout)?,
        })
    }
}

/// Which payload type the caller explicitly asked for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PayloadKind {
    Tensor,
    Image,
    Audio,
}

impl PayloadKind {
    fn parse(spec: &str) -> Result<Self, String> {
        match spec.to_ascii_lowercase().as_str() {
            "tensor" => Ok(PayloadKind::Tensor),
            "image" => Ok(PayloadKind::Image),
            "audio" => Ok(PayloadKind::Audio),
            other => Err(format!(
                "Unknown payloadType '{}' (expected one of: tensor, image, audio)",
                other
            )),
        }
    }
}

/// The explicit payload type plus the metadata needed to build it. There is no
/// type inference here: a bare array is always `PayloadKind::Tensor`.
struct PayloadSpec {
    kind: PayloadKind,
    color_space: Option<String>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    layout: Option<String>,
}

impl Default for PayloadSpec {
    fn default() -> Self {
        Self {
            kind: PayloadKind::Tensor,
            color_space: None,
            sample_rate: None,
            channels: None,
            layout: None,
        }
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
        "hwc" => Ok(ImageLayout::Hwc),
        "chw" => Ok(ImageLayout::Chw),
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

/// Recognized image channel counts mapped to a color space, used only when the
/// caller asks for an image payload but does not name a color space.
fn infer_color_space(channels: usize) -> Option<ColorSpace> {
    match channels {
        1 => Some(ColorSpace::Grayscale),
        3 => Some(ColorSpace::Rgb),
        4 => Some(ColorSpace::Rgba),
        _ => None,
    }
}

fn throw_exception(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("org/morflow/MorflowException", msg);
}

fn map_error_to_exception(env: &mut JNIEnv, err: MorflowError) {
    let msg = match err {
        MorflowError::Io(e) => format!("IO error: {}", e),
        MorflowError::Parse(e) => format!("Parse error: {}", e),
        MorflowError::Compile(e) => format!("Compile error: {}", e),
        MorflowError::Action(e) => format!("Action error: {}", e),
        MorflowError::Execution(e) => format!("Execution error: {}", e),
        MorflowError::TypeMismatch(e) => format!("Type error: {}", e),
    };
    throw_exception(env, &msg);
}

// ---------------------------------------------------------------------------
// JNI Pipeline Lifecycle Native Methods
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeLoad<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    path_jstr: JString<'local>,
) -> jlong {
    let path_str: String = match env.get_string(&path_jstr) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_exception(&mut env, &format!("Invalid path string: {}", e));
            return 0;
        }
    };

    match Morflow::load(Path::new(&path_str)) {
        Ok(pipeline) => Box::into_raw(Box::new(pipeline)) as jlong,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeFromStr<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    src_jstr: JString<'local>,
) -> jlong {
    let src_str: String = match env.get_string(&src_jstr) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_exception(&mut env, &format!("Invalid pipeline source string: {}", e));
            return 0;
        }
    };

    match Morflow::from_str(&src_str) {
        Ok(pipeline) => Box::into_raw(Box::new(pipeline)) as jlong,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeGetParams<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> jobjectArray {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &*(handle as *const MorflowPipeline) };
    let params: Vec<String> = pipeline.params().iter().map(|p| p.name.clone()).collect();

    let string_class = match env.find_class("java/lang/String") {
        Ok(c) => c,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to find String class: {}", e));
            return std::ptr::null_mut();
        }
    };

    let empty_str = match env.new_string("") {
        Ok(s) => s,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to create empty String: {}", e));
            return std::ptr::null_mut();
        }
    };

    let array = match env.new_object_array(params.len() as i32, &string_class, &empty_str) {
        Ok(arr) => arr,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to create String array: {}", e));
            return std::ptr::null_mut();
        }
    };

    for (i, param_name) in params.iter().enumerate() {
        if let Ok(jstr) = env.new_string(param_name) {
            let _ = env.set_object_array_element(&array, i as i32, &jstr);
        }
    }

    array.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeRun<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    input_tensor_obj: JObject<'local>,
) -> jobject {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };

    let input_payload = match java_tensor_to_payload(&mut env, &input_tensor_obj) {
        Ok(p) => p,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to convert input tensor: {}", e));
            return std::ptr::null_mut();
        }
    };

    let outputs = match pipeline.run(input_payload) {
        Ok(out) => out,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let single_payload = match outputs.into_single() {
        Ok(p) => p,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    match payload_to_java_tensor(&mut env, &single_payload) {
        Ok(obj) => obj.into_raw(),
        Err(e) => {
            throw_exception(
                &mut env,
                &format!("Failed to convert output payload: {}", e),
            );
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeRunDirect<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    buffer_obj: JByteBuffer<'local>,
    shape_obj: JIntArray<'local>,
    dtype_jstr: JString<'local>,
    spec: PayloadSpecJava,
) -> jobject {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };

    let spec = match spec.into_rust(&mut env) {
        Ok(s) => s,
        Err(e) => {
            throw_exception(&mut env, &format!("Invalid payload type: {}", e));
            return std::ptr::null_mut();
        }
    };

    let input_payload =
        match direct_to_payload(&mut env, &buffer_obj, &shape_obj, &dtype_jstr, &spec) {
            Ok(p) => p,
            Err(e) => {
                throw_exception(&mut env, &format!("Failed to convert direct buffer: {}", e));
                return std::ptr::null_mut();
            }
        };

    let outputs = match pipeline.run(input_payload) {
        Ok(out) => out,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let single_payload = match outputs.into_single() {
        Ok(p) => p,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    match payload_to_java_tensor(&mut env, &single_payload) {
        Ok(obj) => obj.into_raw(),
        Err(e) => {
            throw_exception(
                &mut env,
                &format!("Failed to convert output payload: {}", e),
            );
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeRunAll<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    input_tensor_obj: JObject<'local>,
) -> jobject {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };

    let input_payload = match java_tensor_to_payload(&mut env, &input_tensor_obj) {
        Ok(p) => p,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to convert input tensor: {}", e));
            return std::ptr::null_mut();
        }
    };

    let outputs = match pipeline.run(input_payload) {
        Ok(out) => out,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let hash_map_class = match env.find_class("java/util/HashMap") {
        Ok(c) => c,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to find HashMap class: {}", e));
            return std::ptr::null_mut();
        }
    };

    let map_obj = match env.new_object(&hash_map_class, "()V", &[]) {
        Ok(m) => m,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to instantiate HashMap: {}", e));
            return std::ptr::null_mut();
        }
    };

    for (name, payload) in outputs.into_iter() {
        let key_jstr = match env.new_string(&name) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let val_tensor = match payload_to_java_tensor(&mut env, &payload) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let _ = env.call_method(
            &map_obj,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&key_jstr), JValue::Object(&val_tensor)],
        );
    }

    map_obj.into_raw()
}

/// Converts one element of a `Object[]` positional-arguments array into a
/// payload. Recognizes numbers, strings, booleans, byte arrays, and
/// `MorflowTensor` instances.
fn java_arg_to_payload<'local>(
    env: &mut JNIEnv<'local>,
    obj: &JObject<'local>,
) -> Result<Payload, String> {
    if obj.is_null() {
        return Ok(Payload::Data {
            buffer: RVec::new(),
        });
    }

    if env
        .is_instance_of(obj, "org/morflow/MorflowTensor")
        .map_err(|e| format!("Failed to inspect argument type: {}", e))?
    {
        return java_tensor_to_payload(env, obj);
    }
    if env
        .is_instance_of(obj, "[B")
        .map_err(|e| format!("Failed to inspect argument type: {}", e))?
    {
        let byte_arr: &JByteArray = obj.into();
        let raw: Vec<u8> = env
            .convert_byte_array(byte_arr)
            .map_err(|e| format!("Failed to read byte[] argument: {}", e))?;
        return Ok(Payload::Data {
            buffer: RVec::from(raw),
        });
    }
    // Numbers, booleans, and strings all carry their text representation.
    for class in ["java/lang/Number", "java/lang/Boolean", "java/lang/String"] {
        if env
            .is_instance_of(obj, class)
            .map_err(|e| format!("Failed to inspect argument type: {}", e))?
        {
            let repr = env
                .call_method(obj, "toString", "()Ljava/lang/String;", &[])
                .map_err(|e| format!("Failed to stringify argument: {}", e))?
                .l()
                .map_err(|e| format!("toString() did not return a String: {}", e))?;
            let text: String = env
                .get_string(&JString::from(repr))
                .map_err(|e| format!("Failed to read argument text: {}", e))?
                .into();
            return Ok(Payload::Data {
                buffer: RVec::from(text.into_bytes()),
            });
        }
    }

    Err(
        "Unsupported argument type (expected a number, string, boolean, byte[], or MorflowTensor)"
            .into(),
    )
}

/// Collects every element of an `Object[]` positional-arguments array.
fn java_args_to_payloads<'local>(
    env: &mut JNIEnv<'local>,
    args_obj: JObject<'local>,
) -> Result<Vec<Payload>, String> {
    if args_obj.is_null() {
        return Ok(Vec::new());
    }
    let array = JObjectArray::from(args_obj);
    let len = env
        .get_array_length(&array)
        .map_err(|e| format!("Failed to read arguments array length: {}", e))?;
    let mut payloads = Vec::with_capacity(len as usize);
    for i in 0..len {
        let element = env
            .get_object_array_element(&array, i)
            .map_err(|e| format!("Failed to read argument {}: {}", i, e))?;
        payloads.push(java_arg_to_payload(env, &element)?);
    }
    Ok(payloads)
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeRunArgs<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    args_obj: JObject<'local>,
) -> jobject {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };

    let payloads = match java_args_to_payloads(&mut env, args_obj) {
        Ok(p) => p,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to convert arguments: {}", e));
            return std::ptr::null_mut();
        }
    };

    let outputs = match pipeline.run_args(payloads) {
        Ok(out) => out,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let single_payload = match outputs.into_single() {
        Ok(p) => p,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    match payload_to_java_tensor(&mut env, &single_payload) {
        Ok(obj) => obj.into_raw(),
        Err(e) => {
            throw_exception(
                &mut env,
                &format!("Failed to convert output payload: {}", e),
            );
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeRunAllArgs<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    args_obj: JObject<'local>,
) -> jobject {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };

    let payloads = match java_args_to_payloads(&mut env, args_obj) {
        Ok(p) => p,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to convert arguments: {}", e));
            return std::ptr::null_mut();
        }
    };

    let outputs = match pipeline.run_args(payloads) {
        Ok(out) => out,
        Err(e) => {
            map_error_to_exception(&mut env, e);
            return std::ptr::null_mut();
        }
    };

    let hash_map_class = match env.find_class("java/util/HashMap") {
        Ok(c) => c,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to find HashMap class: {}", e));
            return std::ptr::null_mut();
        }
    };

    let map_obj = match env.new_object(&hash_map_class, "()V", &[]) {
        Ok(m) => m,
        Err(e) => {
            throw_exception(&mut env, &format!("Failed to instantiate HashMap: {}", e));
            return std::ptr::null_mut();
        }
    };

    for (name, payload) in outputs.into_iter() {
        let key_jstr = match env.new_string(&name) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let val_tensor = match payload_to_java_tensor(&mut env, &payload) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let _ = env.call_method(
            &map_obj,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&key_jstr), JValue::Object(&val_tensor)],
        );
    }

    map_obj.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_morflow_Pipeline_nativeDestroy<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut MorflowPipeline));
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion Helpers (Java <-> Rust Payload)
// ---------------------------------------------------------------------------

fn direct_to_payload<'local>(
    env: &mut JNIEnv<'local>,
    buffer_obj: &JByteBuffer<'local>,
    shape_obj: &JIntArray<'local>,
    dtype_jstr: &JString<'local>,
    spec: &PayloadSpec,
) -> Result<Payload, String> {
    if buffer_obj.is_null() {
        return Ok(Payload::Data {
            buffer: RVec::new(),
        });
    }

    let ptr = env
        .get_direct_buffer_address(buffer_obj)
        .map_err(|e| format!("ByteBuffer must be direct: {}", e))?;
    let capacity = env
        .get_direct_buffer_capacity(buffer_obj)
        .map_err(|e| format!("Failed to get direct buffer capacity: {}", e))?;

    let byte_slice: &[u8] = unsafe { std::slice::from_raw_parts(ptr, capacity) };

    let shape_len = if !shape_obj.is_null() {
        env.get_array_length(shape_obj)
            .map_err(|e| format!("Failed to get shape length: {}", e))? as usize
    } else {
        0
    };

    let mut shape = vec![0i32; shape_len];
    if shape_len > 0 {
        env.get_int_array_region(shape_obj, 0, &mut shape)
            .map_err(|e| format!("Failed to read shape elements: {}", e))?;
    }

    let usize_shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();

    let dtype_str: String = if !dtype_jstr.is_null() {
        env.get_string(dtype_jstr)
            .map(|s| s.into())
            .unwrap_or_else(|_| "raw".to_string())
    } else {
        "raw".to_string()
    };

    construct_payload_from_raw(byte_slice, usize_shape, &dtype_str, spec)
}

fn java_tensor_to_payload<'local>(
    env: &mut JNIEnv<'local>,
    obj: &JObject<'local>,
) -> Result<Payload, String> {
    if obj.is_null() {
        return Ok(Payload::Data {
            buffer: RVec::new(),
        });
    }

    // 1. Get `data` ByteBuffer
    let data_obj = env
        .call_method(obj, "getData", "()Ljava/nio/ByteBuffer;", &[])
        .map_err(|e| format!("Failed to call getData(): {}", e))?
        .l()
        .map_err(|e| format!("getData() did not return Object: {}", e))?;

    if data_obj.is_null() {
        return Ok(Payload::Data {
            buffer: RVec::new(),
        });
    }

    // Direct ByteBuffer zero-copy extraction
    let byte_buffer = JByteBuffer::from(data_obj);

    // 2. Get `shape` int[]
    let shape_obj = env
        .call_method(obj, "getShape", "()[I", &[])
        .map_err(|e| format!("Failed to call getShape(): {}", e))?
        .l()
        .map_err(|e| format!("getShape() did not return Object: {}", e))?;

    let shape_array = JIntArray::from(shape_obj);

    // 3. Get `dtype` String
    let dtype_obj = env
        .call_method(obj, "getDtype", "()Ljava/lang/String;", &[])
        .map_err(|e| format!("Failed to call getDtype(): {}", e))?
        .l()
        .map_err(|e| format!("getDtype() did not return Object: {}", e))?;

    let dtype_jstr = JString::from(dtype_obj);

    // 4. Get the explicit payload type and its optional metadata.
    let spec = read_payload_spec(env, obj)?;

    direct_to_payload(env, &byte_buffer, &shape_array, &dtype_jstr, &spec)
}

/// Reads `getPayloadType()` plus the optional `getColorSpace()`,
/// `getSampleRate()`, `getChannels()`, and `getLayout()` accessors off a Java
/// `MorflowTensor`. A `null` payload type means a plain tensor.
fn read_payload_spec<'local>(
    env: &mut JNIEnv<'local>,
    obj: &JObject<'local>,
) -> Result<PayloadSpec, String> {
    fn call_string<'local>(
        env: &mut JNIEnv<'local>,
        obj: &JObject<'local>,
        method: &str,
    ) -> Result<Option<String>, String> {
        let ret = env
            .call_method(obj, method, "()Ljava/lang/String;", &[])
            .map_err(|e| format!("Failed to call {}(): {}", method, e))?;
        let s = JString::from(
            ret.l()
                .map_err(|e| format!("{}() did not return Object: {}", method, e))?,
        );
        if s.is_null() {
            return Ok(None);
        }
        let v: String = env
            .get_string(&s)
            .map_err(|e| format!("Failed to read {}(): {}", method, e))?
            .into();
        if v.is_empty() {
            return Ok(None);
        }
        Ok(Some(v))
    }

    let kind = match call_string(env, obj, "getPayloadType")? {
        Some(spec) => PayloadKind::parse(&spec)?,
        None => PayloadKind::Tensor,
    };

    let sample_rate = env
        .call_method(obj, "getSampleRate", "()I", &[])
        .map(|v| v.i().unwrap_or(0))
        .ok()
        .filter(|&v| v > 0)
        .map(|v| v as u32);

    let channels = env
        .call_method(obj, "getChannels", "()I", &[])
        .map(|v| v.i().unwrap_or(0))
        .ok()
        .filter(|&v| v > 0)
        .map(|v| v as u32);

    Ok(PayloadSpec {
        kind,
        color_space: call_string(env, obj, "getColorSpace")?,
        sample_rate,
        channels,
        layout: call_string(env, obj, "getLayout")?,
    })
}

fn construct_payload_from_raw(
    byte_slice: &[u8],
    usize_shape: Vec<usize>,
    dtype_str: &str,
    spec: &PayloadSpec,
) -> Result<Payload, String> {
    // Construct Payload from byte slice, shape, and dtype
    let tensor = match dtype_str.to_lowercase().as_str() {
        "f32" | "float" | "float32" => {
            if !byte_slice.len().is_multiple_of(4) {
                return Err("Byte slice length not divisible by 4 for F32 tensor".into());
            }
            let f32_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(byte_slice.as_ptr() as *const f32, byte_slice.len() / 4)
            };
            Tensor::from_f32_shape(f32_slice, usize_shape)?
        }
        "u8" | "uint8" | "byte" => {
            Tensor::from_rvec_u8(RVec::from(byte_slice.to_vec()), usize_shape, TensorDType::U8)?
        }
        "i32" | "int32" => {
            Tensor::from_rvec_u8(RVec::from(byte_slice.to_vec()), usize_shape, TensorDType::I32)?
        }
        _ /* "raw" */ => {
            return Ok(Payload::Data {
                buffer: RVec::from(byte_slice.to_vec()),
            });
        }
    };

    match spec.kind {
        PayloadKind::Tensor => Ok(Payload::Tensor(tensor)),
        PayloadKind::Image => {
            let color_space = match spec.color_space.as_deref() {
                Some(name) => parse_color_space(name)?,
                None => tensor
                    .shape
                    .get(2)
                    .copied()
                    .and_then(infer_color_space)
                    .ok_or_else(|| {
                        format!(
                            "payloadType \"image\" could not infer a color space from shape {:?}; \
                             pass colorSpace=\"grayscale\", \"rgb\", \"rgba\", \"bgr\", or \"bgra\"",
                            tensor.shape.as_slice()
                        )
                    })?,
            };
            let layout = match spec.layout.as_deref() {
                Some(name) => parse_image_layout(name)?,
                None => ImageLayout::Hwc,
            };
            let img = Image::new(tensor, color_space, layout)
                .map_err(|e| format!("Invalid image payload: {}", e))?;
            Ok(Payload::Image(img))
        }
        PayloadKind::Audio => {
            let layout = match spec.layout.as_deref() {
                Some(name) => parse_audio_layout(name)?,
                None => AudioLayout::Planar,
            };
            let channel_layout = match spec.channels {
                Some(ch) => AudioChannelLayout::from_channel_count(ch as usize),
                None => AudioChannelLayout::from_channel_count(
                    tensor.shape.first().copied().unwrap_or(1),
                ),
            };
            let aud = Audio::new(
                tensor,
                spec.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE),
                channel_layout,
                layout,
            )
            .map_err(|e| format!("Invalid audio payload: {}", e))?;
            Ok(Payload::Audio(aud))
        }
    }
}

fn payload_to_java_tensor<'local>(
    env: &mut JNIEnv<'local>,
    payload: &Payload,
) -> Result<JObject<'local>, String> {
    match payload {
        Payload::Tensor(t) => tensor_to_java_morflow_tensor(env, t),
        Payload::Image(img) => tensor_to_java_morflow_tensor(env, &img.tensor),
        Payload::Audio(aud) => tensor_to_java_morflow_tensor(env, &aud.tensor),
        Payload::Data { buffer } => {
            let bytes = buffer.as_slice();
            let shape = vec![bytes.len() as i32];
            create_java_morflow_tensor(env, bytes, &shape, "u8")
        }
        Payload::WithArgs { payload, .. } => payload_to_java_tensor(env, payload),
        Payload::Error(err) => Err(err.to_string()),
        Payload::Composite(_) => {
            Err("Composite payloads cannot be returned directly as a single MorflowTensor".into())
        }
        Payload::Scalar(t) => tensor_to_java_morflow_tensor(env, t),
        Payload::Arg(_) => {
            Err("Argument payloads cannot be returned directly as a MorflowTensor".into())
        }
    }
}

fn tensor_to_java_morflow_tensor<'local>(
    env: &mut JNIEnv<'local>,
    tensor: &Tensor,
) -> Result<JObject<'local>, String> {
    let dtype_str = match tensor.dtype {
        TensorDType::F32 => "f32",
        TensorDType::U8 => "u8",
        TensorDType::I32 => "i32",
        _ => "raw",
    };

    let shape: Vec<i32> = tensor.shape.iter().map(|&d| d as i32).collect();
    if let Some(bytes) = tensor.as_bytes() {
        create_java_morflow_tensor(env, bytes, &shape, dtype_str)
    } else {
        let bytes = tensor.to_contiguous_bytes();
        create_java_morflow_tensor(env, bytes.as_slice(), &shape, dtype_str)
    }
}

fn create_java_morflow_tensor<'local>(
    env: &mut JNIEnv<'local>,
    bytes: &[u8],
    shape: &[i32],
    dtype: &str,
) -> Result<JObject<'local>, String> {
    // 1. Allocate a direct ByteBuffer in Java and fill with bytes
    let byte_buffer_class = env
        .find_class("java/nio/ByteBuffer")
        .map_err(|e| format!("Failed to find ByteBuffer class: {}", e))?;

    let capacity = bytes.len() as i32;
    let direct_buf = env
        .call_static_method(
            &byte_buffer_class,
            "allocateDirect",
            "(I)Ljava/nio/ByteBuffer;",
            &[JValue::Int(capacity)],
        )
        .map_err(|e| format!("Failed to call ByteBuffer.allocateDirect: {}", e))?
        .l()
        .map_err(|e| format!("allocateDirect did not return Object: {}", e))?;

    if !bytes.is_empty() {
        let byte_buf: &JByteBuffer = (&direct_buf).into();
        let dest_ptr = env
            .get_direct_buffer_address(byte_buf)
            .map_err(|e| format!("Failed to get direct buffer destination ptr: {}", e))?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dest_ptr, bytes.len());
        }
    }

    // 2. Create int[] shape array
    let shape_array = env
        .new_int_array(shape.len() as i32)
        .map_err(|e| format!("Failed to allocate shape int[] array: {}", e))?;
    env.set_int_array_region(&shape_array, 0, shape)
        .map_err(|e| format!("Failed to set shape int[] elements: {}", e))?;

    // 3. Create dtype String
    let dtype_jstr = env
        .new_string(dtype)
        .map_err(|e| format!("Failed to create dtype String: {}", e))?;

    // 4. Instantiate org.morflow.MorflowTensor(ByteBuffer, int[], String)
    let morflow_tensor_class = env
        .find_class("org/morflow/MorflowTensor")
        .map_err(|e| format!("Failed to find MorflowTensor class: {}", e))?;

    let tensor_obj = env
        .new_object(
            &morflow_tensor_class,
            "(Ljava/nio/ByteBuffer;[ILjava/lang/String;)V",
            &[
                JValue::Object(&direct_buf),
                JValue::Object(&shape_array),
                JValue::Object(&dtype_jstr),
            ],
        )
        .map_err(|e| format!("Failed to instantiate MorflowTensor: {}", e))?;

    Ok(tensor_obj)
}
