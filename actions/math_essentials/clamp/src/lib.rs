use core_types::{
    ActionArgs, DataType, GetShapeResultFn, Payload, Shape, ShapeResult, TensorDType,
};
use rayon::prelude::*;

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
const _: GetShapeResultFn = get_output_shape_result;

#[no_mangle]
pub extern "C" fn get_output_shape_result(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut min_val = f32::NEG_INFINITY;
    let mut max_val = f32::INFINITY;

    if let Some(args) = &args_opt {
        if let Some(min_str) = args
            .get_named("min")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(m) = min_str.parse::<f32>() {
                min_val = m;
            }
        }
        if let Some(max_str) = args
            .get_named("max")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(m) = max_str.parse::<f32>() {
                max_val = m;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let slice = tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = x.clamp(min_val, max_val));
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'clamp\' requires a tensor or scalar value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_clamp_action() {
        let tensor = Tensor::from_f32_slice(&[-2.0, 0.5, 3.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("min"), RString::from("0.0")));
        named.push(Tuple2(RString::from("max"), RString::from("1.0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 0.5, 1.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_clamp_accepts_a_scalar_value() {
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
