use core_types::{parse_shape_str, ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn reshape_shape(shape_str: &str, total: usize) -> Result<Vec<usize>, core_types::RString> {
    let shape = parse_shape_str(shape_str, total)?;
    if shape.is_empty() {
        return Err("reshape requires a nonempty target shape".into());
    }
    let count = shape
        .iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
        .ok_or_else(|| core_types::RString::from("reshape target element count overflows"))?;
    if count != total {
        return Err(format!(
            "Cannot reshape {total} elements into shape {shape:?} with {count} elements"
        )
        .into());
    }
    Ok(shape)
}

fn shape_impl(input: Shape, args: ActionArgs) -> ShapeResult {
    let shape_str = args
        .get_named("shape")
        .or_else(|| args.positional.first().map(|s| s.as_str()));
    let Some(shape_str) = shape_str else {
        return ShapeResult::Invalid("reshape action requires 'shape' argument".into());
    };
    // The checker forwards unresolved argument values as $name.
    if shape_str.starts_with('$') {
        return ShapeResult::Unknown;
    }
    if input.dims().contains(&0) {
        // Validate syntax and inference rules without inventing an element count.
        return match parse_shape_str(shape_str, 0) {
            Ok(shape) if !shape.is_empty() => {
                if shape
                    .iter()
                    .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                    .is_none()
                {
                    ShapeResult::Invalid("reshape target element count overflows".into())
                } else {
                    ShapeResult::Unknown
                }
            }
            Ok(_) => ShapeResult::Invalid("reshape requires a nonempty target shape".into()),
            Err(error) => ShapeResult::Invalid(error),
        };
    }
    let Some(total) = input
        .dims()
        .iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
    else {
        return ShapeResult::Invalid("reshape input element count overflows".into());
    };
    match reshape_shape(shape_str, total) {
        Ok(shape) => Shape::new(shape).into(),
        Err(error) => ShapeResult::Invalid(error),
    }
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut shape_str = None;
    if let Some(args) = &args_opt {
        shape_str = args
            .get_named("shape")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            let Some(s_str) = shape_str else {
                return Payload::Error("reshape action requires 'shape' argument".into());
            };
            let parsed_shape = match reshape_shape(s_str, tensor.num_elements()) {
                Ok(s) => s,
                Err(e) => return Payload::Error(e),
            };
            match tensor.reshape(parsed_shape) {
                Ok(reshaped) => Payload::from_tensor(reshaped),
                Err(e) => Payload::Error(e),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'reshape\' requires a tensor or scalar value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_reshape_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("shape"), RString::from("[3, 2]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[3, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    fn assert_verdict(actual: ShapeResult, expected: ShapeResult) {
        match (actual, expected) {
            (ShapeResult::Ok(actual), ShapeResult::Ok(expected)) => assert_eq!(actual, expected),
            (ShapeResult::Unknown, ShapeResult::Unknown) => {}
            (actual, expected) => panic!("Expected {expected:?}, got {actual:?}"),
        }
    }

    fn args(shape: Option<&str>) -> ActionArgs {
        let mut args = ActionArgs::default();
        if let Some(shape) = shape {
            args.named.push(Tuple2("shape".into(), shape.into()));
        }
        args
    }

    #[test]
    fn rejects_missing_malformed_and_incompatible_shapes() {
        for target in [
            None,
            Some("[]"),
            Some("[oops]"),
            Some("[-1,-1]"),
            Some("[4,2]"),
            Some("[4,-1]"),
        ] {
            let args = args(target);
            assert!(
                matches!(
                    get_output_shape(Shape::new([2, 3]), args.clone()),
                    ShapeResult::Invalid(_)
                ),
                "{target:?}"
            );
            let input = Tensor::from_f32_shape(&[0.0; 6], vec![2, 3]).unwrap();
            assert!(matches!(
                process(Payload::WithArgs {
                    payload: RBox::new(Payload::Tensor(input)),
                    args
                }),
                Payload::Error(_)
            ));
        }
    }

    #[test]
    fn inferred_and_explicit_shapes_match_execution_including_scalar_input() {
        for (source, target, expected) in [
            (vec![2, 3], "[3,-1]", vec![3, 2]),
            (vec![6], "[2,3]", vec![2, 3]),
            (vec![], "[1]", vec![1]),
        ] {
            let args = args(Some(target));
            assert_verdict(
                get_output_shape(Shape::new(source.clone()), args.clone()),
                ShapeResult::Ok(Shape::new(expected.clone())),
            );
            let input =
                Tensor::from_f32_shape(&vec![0.0; source.iter().product()], source).unwrap();
            let result = process(Payload::WithArgs {
                payload: RBox::new(Payload::from_tensor(input)),
                args,
            });
            match result {
                Payload::Tensor(t) => assert_eq!(t.shape.as_slice(), expected),
                other => panic!("Unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn unknown_dimensions_and_dynamic_targets_are_not_invalid() {
        for target in ["[2,3]", "[2,-1]", "$shape"] {
            assert_verdict(
                get_output_shape(Shape::new([0, 3]), args(Some(target))),
                ShapeResult::Unknown,
            );
        }
        assert!(matches!(
            get_output_shape(Shape::new([0, 3]), args(Some("[-1,-1]"))),
            ShapeResult::Invalid(_)
        ));
        let target = format!("[{},{}]", usize::MAX, usize::MAX);
        assert!(matches!(
            get_output_shape(Shape::new([2, 3]), args(Some(&target))),
            ShapeResult::Invalid(_)
        ));
        assert!(matches!(
            get_output_shape(Shape::new([0, 3]), args(Some(&target))),
            ShapeResult::Invalid(_)
        ));
        let target = format!("[{}, {}, -1]", usize::MAX, usize::MAX);
        assert!(matches!(
            get_output_shape(Shape::new([0, 3]), args(Some(&target))),
            ShapeResult::Invalid(_)
        ));
    }
}
