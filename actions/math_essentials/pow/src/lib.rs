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

    let mut exponent = 1.0f32;
    if let Some(args) = &args_opt {
        if let Some(exp_str) = args
            .get_named("exponent")
            .or_else(|| args.get_named("power"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(exp) = exp_str.parse::<f32>() {
                exponent = exp;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x = x.powf(exponent));
            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from("Action \'pow\' requires Payload::Tensor")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_pow_action() {
        let tensor = Tensor::from_f32_slice(&[2.0, 3.0, 4.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("exponent"), RString::from("2.0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[4.0, 9.0, 16.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
