use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult, TensorDType};
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
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub static MORFLOW_SHAPE_ABI: u32 = core_types::SHAPE_ABI_VERSION;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut alpha = 0.01f32;
    if let Some(args) = &args_opt {
        if let Some(a_str) = args
            .get_named("alpha")
            .or_else(|| args.get_named("negative_slope"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(a) = a_str.parse::<f32>() {
                alpha = a;
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
                .for_each(|x| *x = if *x >= 0.0 { *x } else { *x * alpha });
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'leaky_relu\' requires a tensor or scalar value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_leaky_relu_action() {
        let tensor = Tensor::from_f32_slice(&[-2.0, 0.0, 3.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("alpha"), RString::from("0.1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[-0.2, 0.0, 3.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_leaky_relu_accepts_a_scalar_value() {
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
