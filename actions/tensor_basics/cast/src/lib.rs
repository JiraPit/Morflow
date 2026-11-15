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
    // An unknown target dtype can never run.
    if let Ok(Some(name)) = core_types::contract::value(&args, &["dtype"], Some(0)) {
        if let Err(reason) = target_dtype(name) {
            return ShapeResult::Invalid(reason.into());
        }
    }
    shape_impl(input, args).into()
}

/// The dtype named by `name`: any dtype name, plus the "int", "short" and
/// "long" spellings for `i32`, `i16` and `i64`.
fn target_dtype(name: &str) -> Result<TensorDType, String> {
    match name.to_ascii_lowercase().as_str() {
        "int" => Ok(TensorDType::I32),
        "short" => Ok(TensorDType::I16),
        "long" => Ok(TensorDType::I64),
        other => TensorDType::from_name(other).ok_or_else(|| {
            format!(
                "Unknown cast dtype '{}' (expected one of: f32, f64, u8, i8, i16, i32, i64, u32, u64)",
                name
            )
        }),
    }
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut dtype = TensorDType::F32;
    if let Some(args) = &args_opt {
        if let Some(dt_str) = args
            .get_named("dtype")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            dtype = match target_dtype(dt_str) {
                Ok(dtype) => dtype,
                Err(reason) => return Payload::Error(reason.into()),
            };
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match tensor.cast(dtype) {
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

    fn cast_to(tensor: Tensor, dtype: &str) -> Payload {
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dtype"), RString::from(dtype)));
        process(Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        })
    }

    fn cast_tensor(tensor: Tensor, dtype: &str) -> Tensor {
        match cast_to(tensor, dtype) {
            Payload::Tensor(out) => out,
            other => panic!(
                "cast to {dtype} produced {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_cast_reaches_every_dtype() {
        let source = Tensor::from_vec_of(vec![1i32, -2, 3], vec![3]).unwrap();
        for (name, dtype) in [
            ("f32", TensorDType::F32),
            ("f64", TensorDType::F64),
            ("u8", TensorDType::U8),
            ("i8", TensorDType::I8),
            ("i16", TensorDType::I16),
            ("i32", TensorDType::I32),
            ("i64", TensorDType::I64),
            ("u32", TensorDType::U32),
            ("u64", TensorDType::U64),
        ] {
            let out = cast_tensor(source.clone(), name);
            assert_eq!(out.dtype, dtype, "cast to {name}");
            assert_eq!(out.shape.as_slice(), &[3]);
        }
    }

    #[test]
    fn test_cast_is_exact_without_going_through_f32() {
        let big = (1i64 << 40) + 1;
        let out = cast_tensor(Tensor::from_vec_of(vec![big, -3], vec![2]).unwrap(), "f64");
        assert_eq!(out.to_vec_of::<f64>().unwrap(), vec![big as f64, -3.0]);

        let out = cast_tensor(
            Tensor::from_vec_of(vec![16_777_217i32], vec![1]).unwrap(),
            "i64",
        );
        assert_eq!(out.to_vec_of::<i64>().unwrap(), vec![16_777_217]);

        let out = cast_tensor(Tensor::from_vec_of(vec![0.1f64], vec![1]).unwrap(), "f32");
        assert_eq!(out.to_vec_of::<f32>().unwrap(), vec![0.1f32]);
    }

    #[test]
    fn test_cast_rounds_floats_and_clamps_to_the_target_range() {
        let floats = Tensor::from_vec_of(vec![2.5f32, -1.4, 300.0, f32::NAN], vec![4]).unwrap();
        let out = cast_tensor(floats.clone(), "u8");
        assert_eq!(out.to_vec_of::<u8>().unwrap(), vec![3, 0, 255, 0]);
        let out = cast_tensor(floats, "i8");
        assert_eq!(out.to_vec_of::<i8>().unwrap(), vec![3, -1, 127, 0]);

        let ints = Tensor::from_vec_of(vec![-5i64, 70_000], vec![2]).unwrap();
        let out = cast_tensor(ints.clone(), "u32");
        assert_eq!(out.to_vec_of::<u32>().unwrap(), vec![0, 70_000]);
        let out = cast_tensor(ints, "i16");
        assert_eq!(out.to_vec_of::<i16>().unwrap(), vec![-5, i16::MAX]);
    }

    #[test]
    fn test_cast_accepts_long_names_and_older_spellings() {
        let source = Tensor::from_vec_of(vec![7u8], vec![1]).unwrap();
        for (name, dtype) in [
            ("uint64", TensorDType::U64),
            ("float64", TensorDType::F64),
            ("long", TensorDType::I64),
            ("short", TensorDType::I16),
            ("int", TensorDType::I32),
            ("double", TensorDType::F64),
        ] {
            assert_eq!(
                cast_tensor(source.clone(), name).dtype,
                dtype,
                "cast to {name}"
            );
        }
    }

    #[test]
    fn test_cast_rejects_an_unknown_dtype() {
        let source = Tensor::from_vec_of(vec![1.0f32], vec![1]).unwrap();
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dtype"), RString::from("f16")));
        let args = ActionArgs {
            positional: core_types::RVec::new(),
            named,
        };
        let payload = Payload::Tensor(source.clone());
        match shapecheck(core_types::InputDescriptor::from_payload(&payload), args) {
            core_types::ShapeCheckResult::Invalid { reason } => {
                assert!(reason.contains("Unknown cast dtype 'f16'"), "{reason}");
            }
            _ => panic!("an unknown dtype should be rejected by the shape check"),
        }
        match cast_to(source, "f16") {
            Payload::Error(reason) => assert!(reason.contains("Unknown cast dtype"), "{reason}"),
            other => panic!(
                "expected an error, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
