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
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = if *x >= 0.0 { *x } else { *x * alpha });
            Payload::Tensor(tensor)
        }
        Payload::Image(mut image) if image.dtype() == TensorDType::F32 => {
            let slice = image.tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = if *x >= 0.0 { *x } else { *x * alpha });
            Payload::Image(image)
        }
        Payload::Audio(mut audio) if audio.dtype() == TensorDType::F32 => {
            let slice = audio.tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = if *x >= 0.0 { *x } else { *x * alpha });
            Payload::Audio(audio)
        }
        other => other,
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
}
