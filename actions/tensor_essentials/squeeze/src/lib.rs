use core_types::{DataType, Payload};

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

    let mut axis = None;
    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<isize>() {
                axis = Some(ax);
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match tensor.squeeze(axis) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Image(image) => match image.tensor.squeeze(axis) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Audio(audio) => match audio.tensor.squeeze(axis) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_squeeze_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![1, 4, 1]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor.clone())),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[4, 1]);
        } else {
            panic!("Expected Tensor output");
        }

        // Squeeze axis -1
        let mut named_neg = core_types::RVec::new();
        named_neg.push(Tuple2(RString::from("axis"), RString::from("-1")));
        let payload_neg = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor.clone())),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named: named_neg,
            },
        };

        let res_neg = process(payload_neg);
        if let Payload::Tensor(out) = res_neg {
            assert_eq!(out.shape.as_slice(), &[1, 4]);
        } else {
            panic!("Expected Tensor output");
        }

        // Squeeze all
        let res_all = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res_all {
            assert_eq!(out.shape.as_slice(), &[4]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
