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
            slice
                .par_iter_mut()
                .for_each(|x| *x = if *x > 0.0 { x.sqrt() } else { 0.0 });
            Payload::Tensor(tensor)
        }
        Payload::Image(mut image) if image.dtype() == TensorDType::F32 => {
            let slice = image.tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = if *x > 0.0 { x.sqrt() } else { 0.0 });
            Payload::Image(image)
        }
        Payload::Audio(mut audio) if audio.dtype() == TensorDType::F32 => {
            let slice = audio.tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = if *x > 0.0 { x.sqrt() } else { 0.0 });
            Payload::Audio(audio)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_sqrt_action() {
        let tensor = Tensor::from_f32_slice(&[4.0, 9.0, 16.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[2.0, 3.0, 4.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
