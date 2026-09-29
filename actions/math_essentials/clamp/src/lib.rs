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
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = x.clamp(min_val, max_val));
            Payload::Tensor(tensor)
        }
        Payload::Image(mut image) if image.dtype() == TensorDType::F32 => {
            let slice = image.tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = x.clamp(min_val, max_val));
            Payload::Image(image)
        }
        Payload::Audio(mut audio) if audio.dtype() == TensorDType::F32 => {
            let slice = audio.tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = x.clamp(min_val, max_val));
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
}
