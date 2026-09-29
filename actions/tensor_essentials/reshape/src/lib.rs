use core_types::{parse_shape_str, DataType, Payload};

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

    let mut shape_str = None;
    if let Some(args) = &args_opt {
        shape_str = args
            .get_named("shape")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    match inner_payload {
        Payload::Tensor(tensor) => {
            let Some(s_str) = shape_str else {
                return Payload::Error("reshape action requires 'shape' argument".into());
            };
            let parsed_shape = match parse_shape_str(s_str, tensor.num_elements()) {
                Ok(s) => s,
                Err(e) => return Payload::Error(e),
            };
            match tensor.reshape(parsed_shape) {
                Ok(reshaped) => Payload::Tensor(reshaped),
                Err(e) => Payload::Error(e),
            }
        }
        Payload::Image(image) => {
            let Some(s_str) = shape_str else {
                return Payload::Error("reshape action requires 'shape' argument".into());
            };
            let parsed_shape = match parse_shape_str(s_str, image.tensor.num_elements()) {
                Ok(s) => s,
                Err(e) => return Payload::Error(e),
            };
            match image.tensor.reshape(parsed_shape) {
                Ok(reshaped) => Payload::Tensor(reshaped),
                Err(e) => Payload::Error(e),
            }
        }
        Payload::Audio(audio) => {
            let Some(s_str) = shape_str else {
                return Payload::Error("reshape action requires 'shape' argument".into());
            };
            let parsed_shape = match parse_shape_str(s_str, audio.tensor.num_elements()) {
                Ok(s) => s,
                Err(e) => return Payload::Error(e),
            };
            match audio.tensor.reshape(parsed_shape) {
                Ok(reshaped) => Payload::Tensor(reshaped),
                Err(e) => Payload::Error(e),
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_reshape_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("shape"), RString::from("[3, 2]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[3, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
