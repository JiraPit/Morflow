use core_types::Payload;
use core_types::Tensor;
pub fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    if let Err(error) = crate::require() {
        return Payload::Error(error);
    }
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut shift = 0isize;
    let axis = prepared.unsigned("axis").unwrap_or(0) as usize;

    if let Some(args) = &args_opt {
        if let Some(sh_str) = args
            .get_named("shift")
            .or_else(|| args.get_named("shifts"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(sh) = prepared.args.parse::<isize>(sh_str) {
                shift = sh;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match roll_tensor(&tensor, shift, axis) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'roll\' requires a tensor or scalar value",
        )),
    }
}

fn roll_tensor(tensor: &Tensor, shift: isize, axis: usize) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r == 0 {
        return Ok(tensor.clone());
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

    crate::concat(&[part1, part2], axis as isize).map_err(|e| e.to_string())
}
