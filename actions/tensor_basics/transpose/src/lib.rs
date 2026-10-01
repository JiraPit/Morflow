use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg, axis};
    let rank = input.rank();
    let result = contract::finish((|| {
        let d0 = arg::<isize>(&args, &["dim0"], Some(0), Some(0))?.unwrap();
        let d1 = arg::<isize>(&args, &["dim1"], Some(1), Some(1))?.unwrap();
        let d0 = axis(d0, input.rank(), false)?;
        let d1 = axis(d1, input.rank(), false)?;
        let mut out = input.dims().to_vec();
        out.swap(d0, d1);
        contract::shape(out)
    })());
    match result {
        ShapeResult::Unknown => ShapeResult::Ok(Shape::unknown(rank)),
        other => other,
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
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut dim0 = 0isize;
    let mut dim1 = 1isize;

    if let Some(args) = &args_opt {
        if let Some(d0_str) = args
            .get_named("dim0")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(d0) = prepared.args.parse::<isize>(d0_str) {
                dim0 = d0;
            }
        }
        if let Some(d1_str) = args
            .get_named("dim1")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(d1) = prepared.args.parse::<isize>(d1_str) {
                dim1 = d1;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match tensor.transpose(dim0, dim1) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e),
        },
        _ => Payload::Error(core_types::RString::from(
            "Action \'transpose\' requires a tensor or scalar value",
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
    fn test_transpose_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dim0"), RString::from("0")));
        named.push(Tuple2(RString::from("dim1"), RString::from("1")));
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
            assert!(!out.is_contiguous());
            assert_eq!(out.to_vec_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }

        // Transpose with negative dims (-2, -1)
        let tensor2 = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut named_neg = core_types::RVec::new();
        named_neg.push(Tuple2(RString::from("dim0"), RString::from("-2")));
        named_neg.push(Tuple2(RString::from("dim1"), RString::from("-1")));
        let payload_neg = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor2)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named: named_neg,
            },
        };

        let res_neg = process(payload_neg);
        if let Payload::Tensor(out) = res_neg {
            assert_eq!(out.shape.as_slice(), &[3, 2]);
            assert_eq!(out.to_vec_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
