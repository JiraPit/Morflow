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

    let mut start_dim = 0usize;
    let mut end_dim = -1isize;

    if let Some(args) = &args_opt {
        if let Some(s_str) = args
            .get_named("start_dim")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(s) = s_str.parse::<usize>() {
                start_dim = s;
            }
        }
        if let Some(e_str) = args
            .get_named("end_dim")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(e) = e_str.parse::<isize>() {
                end_dim = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match tensor.flatten(start_dim, end_dim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        _ => Payload::Error(core_types::RString::from("Action \'flatten\' requires Payload::Tensor")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_flatten_action() {
        let tensor =
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
                .unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("start_dim"), RString::from("1")));
        named.push(Tuple2(RString::from("end_dim"), RString::from("-1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 4]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
