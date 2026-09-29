use core_types::{DataType, Payload, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x = silu(*x));
            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from("Action \'silu\' requires Payload::Tensor")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_silu_action() {
        let tensor = Tensor::from_f32_slice(&[0.0, 1.0, -1.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!(slice[0].abs() < 1e-4);
            assert!((slice[1] - 0.7310586).abs() < 1e-3);
            assert!((slice[2] - (-0.2689414)).abs() < 1e-3);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
