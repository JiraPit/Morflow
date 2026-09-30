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
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| {
                *x = if *x > 0.0 {
                    1.0
                } else if *x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            });
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'sign\' requires a tensor or scalar value",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_sign_action() {
        let tensor = Tensor::from_f32_slice(&[-5.0, 0.0, 10.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[-1.0, 0.0, 1.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_sign_accepts_a_scalar_value() {
        let res = process(Payload::scalar_f32(2.0));
        match res {
            Payload::Scalar(out) => {
                assert_eq!(out.as_f32_slice().unwrap(), &[1.0]);
            }
            other => panic!(
                "scalar path produced the wrong payload: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
