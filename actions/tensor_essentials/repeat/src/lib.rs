use core_types::{DataType, Payload, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn parse_repeats_str(s: &str) -> Result<Vec<usize>, String> {
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
                .map_err(|e| format!("Invalid repeat '{}': {}", p, e))
        })
        .collect()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut repeats_str = None;
    if let Some(args) = &args_opt {
        repeats_str = args
            .get_named("repeats")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    let Some(r_str) = repeats_str else {
        return Payload::Error("repeat action requires 'repeats' argument".into());
    };

    let repeats = match parse_repeats_str(r_str) {
        Ok(r) => r,
        Err(e) => return Payload::Error(e.into()),
    };

    match inner_payload {
        Payload::Tensor(tensor) => match repeat_tensor(&tensor, &repeats) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error(core_types::RString::from("Action \'repeat\' requires Payload::Tensor")),
    }
}

fn repeat_tensor(tensor: &Tensor, repeats: &[usize]) -> Result<Tensor, String> {
    let mut current = tensor.clone();
    let r = current.rank();

    // Pad rank if repeats has more dimensions
    let full_repeats = if repeats.len() < r {
        let mut rep = vec![1; r - repeats.len()];
        rep.extend_from_slice(repeats);
        rep
    } else {
        while current.rank() < repeats.len() {
            current = current.unsqueeze(0).map_err(|e| e.to_string())?;
        }
        repeats.to_vec()
    };

    for (axis, &rep) in full_repeats.iter().enumerate() {
        if rep > 1 {
            let clones: Vec<Tensor> = vec![current.clone(); rep];
            current = Tensor::concat(&clones, axis as isize).map_err(|e| e.to_string())?;
        }
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_repeat_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0], vec![1, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("repeats"), RString::from("[2, 1]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
