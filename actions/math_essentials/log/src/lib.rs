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

    let mut base = "e";
    let mut eps = 1e-8f32;

    if let Some(args) = &args_opt {
        if let Some(b) = args
            .get_named("base")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            base = b;
        }
        if let Some(e_str) = args.get_named("eps") {
            if let Ok(e) = e_str.parse::<f32>() {
                eps = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let slice = tensor.as_f32_slice_mut();
            match base {
                "2" => slice
                    .par_iter_mut()
                    .for_each(|x| *x = (*x + eps).max(eps).log2()),
                "10" => slice
                    .par_iter_mut()
                    .for_each(|x| *x = (*x + eps).max(eps).log10()),
                _ => slice
                    .par_iter_mut()
                    .for_each(|x| *x = (*x + eps).max(eps).ln()),
            }
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'log\' requires a tensor or scalar value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_log_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, std::f32::consts::E]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!(slice[0].abs() < 1e-4);
            assert!((slice[1] - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_log_accepts_a_scalar_value() {
        let res = process(Payload::scalar_f32(2.0));
        match res {
            Payload::Scalar(out) => {
                assert_eq!(out.as_f32_slice().unwrap()[0], (2.0f32 + 1e-8).ln());
            }
            other => panic!(
                "scalar path produced the wrong payload: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
