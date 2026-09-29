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

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x = -*x);
            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from("Action \'neg\' requires Payload::Tensor")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_neg_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, -2.5, 3.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[-1.0, 2.5, -3.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
