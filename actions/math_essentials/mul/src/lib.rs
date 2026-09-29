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

    let mut mul_val = 1.0f32;
    if let Some(args) = &args_opt {
        if let Some(v_str) = args
            .get_named("value")
            .or_else(|| args.get_named("val"))
            .or_else(|| args.get_named("scalar"))
            .or_else(|| args.get_named("factor"))
            .or_else(|| args.get_named("amount"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(v) = v_str.parse::<f32>() {
                mul_val = v;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x *= mul_val);
            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from("Action \'mul\' requires Payload::Tensor")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_mul_action() {
        let tensor = Tensor::from_f32_slice(&[2.0, 4.0, 6.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("value"), RString::from("0.5")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
