use core_types::shapecheck::PreparedArgs;
use core_types::{parse_shape_str, DataType, Payload, Shape, ShapeResult};
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

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
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
    if input.element_count().is_none() && input.dims().contains(&core_types::Dimension::Unknown) {
        let validated = match parse_shape_str(shape_str, 0) {
            Ok(shape) if !shape.is_empty() => shape,
            Ok(_) => {
                return ShapeResult::Invalid("reshape requires a nonempty target shape".into())
            }
            Err(reason) => return ShapeResult::Invalid(reason),
        };
        let raw = shape_str
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_start_matches('(')
            .trim_end_matches(')');
        let target = raw
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .zip(validated)
            .map(|(text, n)| {
                if text == "-1" {
                    core_types::Dimension::Unknown
                } else {
                    n.into()
                }
            });
        return core_types::contract::finish(core_types::contract::shape(target));
    }
    let Some(total) = input.element_count() else {
        return ShapeResult::Invalid("reshape input element count overflows".into());
    };
    match reshape_shape(shape_str, total) {
        Ok(shape) => Shape::new(shape).into(),
        Err(error) => ShapeResult::Invalid(error),
    }
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let inner_payload = payload.into_unwrapped();
    let parsed_shape = match prepared.output_dims() {
        Ok(s) => s,
        Err(e) => return Payload::Error(e),
    };
    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match tensor.reshape(parsed_shape) {
            Ok(reshaped) => Payload::from_tensor(reshaped),
            Err(e) => Payload::Error(e),
        },
        _ => Payload::Error(core_types::RString::from(
            "Action \'reshape\' requires a tensor or scalar value",
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
        for (target, expected) in [
            ("[2,3]", Shape::new([2, 3])),
            (
                "[2,-1]",
                Shape::new([
                    core_types::Dimension::Known(2),
                    core_types::Dimension::Unknown,
                ]),
            ),
        ] {
            assert_verdict(
                get_output_shape(Shape::unknown(2), args(Some(target))),
                ShapeResult::Ok(expected),
            );
        }
        assert!(matches!(
            get_output_shape(Shape::unknown(2), args(Some("$shape"))),
            ShapeResult::Unknown
        ));
        assert!(matches!(
            get_output_shape(
                Shape::new([
                    core_types::Dimension::Unknown,
                    core_types::Dimension::Known(3)
                ]),
                args(Some("[-1,-1]"))
            ),
            ShapeResult::Invalid(_)
        ));
        let target = format!("[{},{}]", usize::MAX, usize::MAX);
        assert!(matches!(
            get_output_shape(Shape::new([2, 3]), args(Some(&target))),
            ShapeResult::Invalid(_)
        ));
        assert!(matches!(
            get_output_shape(
                Shape::new([
                    core_types::Dimension::Unknown,
                    core_types::Dimension::Known(3)
                ]),
                args(Some(&target))
            ),
            ShapeResult::Invalid(_)
        ));
        let target = format!("[{}, {}, -1]", usize::MAX, usize::MAX);
        assert!(matches!(
            get_output_shape(
                Shape::new([
                    core_types::Dimension::Unknown,
                    core_types::Dimension::Known(3)
                ]),
                args(Some(&target))
            ),
            ShapeResult::Invalid(_)
        ));
    }
}
