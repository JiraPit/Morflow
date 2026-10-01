use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, TensorDType};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> Shape {
    input
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

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

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
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
        let payload = Payload::scalar_f32(2.0);
        let core_types::ShapeCheckResult::Ready { prepared, .. } = shapecheck(
            core_types::InputDescriptor::from_payload(&payload),
            core_types::ActionArgs::default(),
        ) else {
            panic!("expected ready")
        };
        let res = crate::process(payload, prepared);
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
