use core_types::{DataType, Payload, Tensor};

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

    let mut shift = 0isize;
    let mut axis = 0usize;

    if let Some(args) = &args_opt {
        if let Some(sh_str) = args
            .get_named("shift")
            .or_else(|| args.get_named("shifts"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(sh) = sh_str.parse::<isize>() {
                shift = sh;
            }
        }
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<usize>() {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match roll_tensor(&tensor, shift, axis) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Image(image) => match roll_tensor(&image.tensor, shift, axis) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Audio(audio) => match roll_tensor(&audio.tensor, shift, axis) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn roll_tensor(tensor: &Tensor, shift: isize, axis: usize) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r == 0 {
        return Ok(tensor.clone());
    }
    if axis >= r {
        return Err(format!(
            "Axis {} out of bounds for tensor of rank {}",
            axis, r
        ));
    }
    let dim_len = tensor.shape[axis];
    if dim_len == 0 {
        return Ok(tensor.clone());
    }

    let shift_norm = ((shift % dim_len as isize) + dim_len as isize) as usize % dim_len;
    if shift_norm == 0 {
        return Ok(tensor.clone());
    }

    let split_idx = dim_len - shift_norm;
    let part1 = tensor
        .slice_range(axis, split_idx, dim_len, 1)
        .map_err(|e| e.to_string())?;
    let part2 = tensor
        .slice_range(axis, 0, split_idx, 1)
        .map_err(|e| e.to_string())?;

    Tensor::concat(&[part1, part2], axis).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_roll_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("shift"), RString::from("2")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[4.0, 5.0, 1.0, 2.0, 3.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
