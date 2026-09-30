use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut eps = 1e-5f32;
    if let Some(args) = &args_opt {
        if let Some(e_str) = args
            .get_named("eps")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(e) = e_str.parse::<f32>() {
                eps = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let r = tensor.rank();
            if r == 0 {
                return Payload::from_tensor(tensor);
            }
            let feat_dim = *tensor.shape.last().unwrap();
            if feat_dim == 0 {
                return Payload::from_tensor(tensor);
            }

            let slice = tensor.as_f32_slice_mut();
            slice.par_chunks_mut(feat_dim).for_each(|feat| {
                let mean_sq: f32 = feat.iter().map(|&x| x * x).sum::<f32>() / feat_dim as f32;
                let inv_rms = 1.0 / (mean_sq + eps).sqrt();
                for x in feat {
                    *x *= inv_rms;
                }
            });
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'rms_norm\' requires a tensor or scalar value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_rms_norm_action() {
        let tensor = Tensor::from_f32_shape(&[3.0, 4.0], vec![1, 2]).unwrap();
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            let rms = 12.5f32.sqrt();
            assert!((slice[0] - (3.0 / rms)).abs() < 1e-3);
            assert!((slice[1] - (4.0 / rms)).abs() < 1e-3);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_rms_norm_accepts_a_scalar_value() {
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
