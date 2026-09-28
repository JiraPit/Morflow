use std::path::Path;

use abi_stable::std_types::RVec;
use core_types::{
    Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
    TensorDType,
};
use jni::objects::{JByteBuffer, JClass, JIntArray, JObject, JString, JValue};
use jni::sys::{jlong, jobject, jobjectArray};
use jni::JNIEnv;
use pipeline::{Morflow, MorflowError, MorflowPipeline};

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
pub extern "system" fn Java_org_morflow_Pipeline_nativeWarmup<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return;
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };
    if let Err(e) = pipeline.warmup() {
        map_error_to_exception(&mut env, e);
    }
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
) -> jobject {
    if handle == 0 {
        throw_exception(&mut env, "Pipeline handle is null");
        return std::ptr::null_mut();
    }
    let pipeline = unsafe { &mut *(handle as *mut MorflowPipeline) };

    let input_payload = match direct_to_payload(&mut env, &buffer_obj, &shape_obj, &dtype_jstr) {
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

    construct_payload_from_raw(byte_slice, usize_shape, &dtype_str)
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

    direct_to_payload(env, &byte_buffer, &shape_array, &dtype_jstr)
}

fn construct_payload_from_raw(
    byte_slice: &[u8],
    usize_shape: Vec<usize>,
    dtype_str: &str,
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

    // Auto-detect Image / Audio if shape matches
    if tensor.rank() == 3 {
        let channels = tensor.shape[2];
        let cs = match channels {
            1 => Some(ColorSpace::Grayscale),
            3 => Some(ColorSpace::Rgb),
            4 => Some(ColorSpace::Rgba),
            _ => None,
        };
        if let Some(color_space) = cs {
            if let Ok(img) = Image::new(tensor.clone(), color_space, ImageLayout::Hwc) {
                return Ok(Payload::Image(img));
            }
        }
    } else if tensor.rank() == 2 && tensor.shape[0] <= 8 {
        let ch = tensor.shape[0];
        let layout = AudioChannelLayout::from_channel_count(ch);
        if let Ok(aud) = Audio::new(tensor.clone(), 44100, layout, AudioLayout::Planar) {
            return Ok(Payload::Audio(aud));
        }
    }

    Ok(Payload::Tensor(tensor))
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
