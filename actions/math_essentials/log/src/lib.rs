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
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
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
            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from("Action \'log\' requires Payload::Tensor")),
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
}
