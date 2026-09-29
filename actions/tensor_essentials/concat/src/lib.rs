use core_types::{DataType, Payload, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut axis = 0isize;
    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<isize>() {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Composite(items) => {
            let mut tensors = Vec::new();
            for item in items {
                match item {
                    Payload::Tensor(t) => tensors.push(t),
                    Payload::Image(img) => tensors.push(img.tensor),
                    Payload::Audio(aud) => tensors.push(aud.tensor),
                    _ => {
                        return Payload::Error(
                            "All items in composite payload must be tensors for concat".into(),
                        )
                    }
                }
            }
            match Tensor::concat(&tensors, axis) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e),
            }
        }
        Payload::Tensor(t) => Payload::Tensor(t),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, RVec, Tensor, Tuple2};

    #[test]
    fn test_concat_action() {
        let t1 = Tensor::from_f32_shape(&[1.0, 2.0], vec![1, 2]).unwrap();
        let t2 = Tensor::from_f32_shape(&[3.0, 4.0], vec![1, 2]).unwrap();

        let mut items = RVec::new();
        items.push(Payload::Tensor(t1));
        items.push(Payload::Tensor(t2));

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Composite(items)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
