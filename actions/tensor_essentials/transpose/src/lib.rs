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

    let mut dim0 = 0usize;
    let mut dim1 = 1usize;

    if let Some(args) = &args_opt {
        if let Some(d0_str) = args
            .get_named("dim0")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(d0) = d0_str.parse::<usize>() {
                dim0 = d0;
            }
        }
        if let Some(d1_str) = args
            .get_named("dim1")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(d1) = d1_str.parse::<usize>() {
                dim1 = d1;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match tensor.transpose(dim0, dim1) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Image(image) => match image.tensor.transpose(dim0, dim1) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Audio(audio) => match audio.tensor.transpose(dim0, dim1) {
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
    fn test_transpose_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dim0"), RString::from("0")));
        named.push(Tuple2(RString::from("dim1"), RString::from("1")));
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
            assert!(!out.is_contiguous());
            assert_eq!(out.to_vec_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
