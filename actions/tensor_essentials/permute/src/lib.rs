use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn parse_dims_str(s: &str) -> Result<Vec<usize>, String> {
    let clean = s
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_start_matches('(')
        .trim_end_matches(')');
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    clean
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| format!("Invalid dimension '{}': {}", p, e))
        })
        .collect()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut dims_str = None;
    if let Some(args) = &args_opt {
        dims_str = args
            .get_named("dims")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    let Some(d_str) = dims_str else {
        return Payload::Error("permute action requires 'dims' argument".into());
    };

    let dims = match parse_dims_str(d_str) {
        Ok(d) => d,
        Err(e) => return Payload::Error(e.into()),
    };

    match inner_payload {
        Payload::Tensor(tensor) => match tensor.permute(&dims) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Image(image) => match image.tensor.permute(&dims) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e),
        },
        Payload::Audio(audio) => match audio.tensor.permute(&dims) {
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
    fn test_permute_action() {
        let tensor =
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
                .unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dims"), RString::from("[2, 0, 1]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2, 2]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
