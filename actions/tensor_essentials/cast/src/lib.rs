use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult, TensorDType};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
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
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match tensor.cast(target_dtype) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e),
        },
        _ => Payload::Error(core_types::RString::from(
            "Action \'cast\' requires a tensor or scalar value",
        )),
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

    #[test]
    fn test_cast_accepts_a_scalar_value() {
        let res = process(Payload::scalar_f32(2.0));
        match res {
            Payload::Scalar(out) => {
                assert_eq!(out.as_f32_slice().unwrap(), &[2.0]);
            }
            other => panic!(
                "scalar path produced the wrong payload: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
