use core_types::{DataType, Payload, TensorDType};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut target_dtype = TensorDType::F32;
    if let Some(args) = &args_opt {
        if let Some(dt_str) = args
            .get_named("dtype")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            match dt_str.to_lowercase().as_str() {
                "f32" | "float" | "float32" => target_dtype = TensorDType::F32,
                "f64" | "double" | "float64" => target_dtype = TensorDType::F64,
                "u8" | "uint8" | "byte" => target_dtype = TensorDType::U8,
                "i16" | "int16" | "short" => target_dtype = TensorDType::I16,
                "i32" | "int" | "int32" => target_dtype = TensorDType::I32,
                "i64" | "long" | "int64" => target_dtype = TensorDType::I64,
                _ => {}
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match tensor.cast(target_dtype) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Image(image) => match image.tensor.cast(target_dtype) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Audio(audio) => match audio.tensor.cast(target_dtype) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_cast_action() {
        let tensor = Tensor::from_f32_slice(&[10.0, 20.0, 30.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dtype"), RString::from("u8")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.dtype, TensorDType::U8);
            assert_eq!(out.as_u8_slice().unwrap(), &[10, 20, 30]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
