use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: ActionArgs) -> ShapeResult {
    use core_types::contract::{self, arg, axis};
    contract::finish((|| {
        let d0 = arg::<isize>(&args, &["dim0"], Some(0), Some(0))?.unwrap();
        let d1 = arg::<isize>(&args, &["dim1"], Some(1), Some(1))?.unwrap();
        let d0 = axis(d0, input.rank(), false)?;
        let d1 = axis(d1, input.rank(), false)?;
        let mut out = input.dims().to_vec();
        out.swap(d0, d1);
        contract::shape(out)
    })())
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

    let mut dim0 = 0isize;
    let mut dim1 = 1isize;

    if let Some(args) = &args_opt {
        if let Some(d0_str) = args
            .get_named("dim0")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(d0) = d0_str.parse::<isize>() {
                dim0 = d0;
            }
        }
        if let Some(d1_str) = args
            .get_named("dim1")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(d1) = d1_str.parse::<isize>() {
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

#[cfg(test)]
mod tests {
    use super::*;
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
