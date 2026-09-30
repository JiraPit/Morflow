use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult, Tensor};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut shift = 0isize;
    let mut axis = 0isize;

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
            if let Ok(ax) = ax_str.parse::<isize>() {
                axis = ax;
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

fn roll_tensor(tensor: &Tensor, shift: isize, raw_ax: isize) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r == 0 {
        return Ok(tensor.clone());
    }
    let axis_idx = if raw_ax < 0 {
        raw_ax + r as isize
    } else {
        raw_ax
    };
    if axis_idx < 0 || axis_idx as usize >= r {
        return Err(format!(
            "Axis {} out of bounds for tensor of rank {}",
            raw_ax, r
        ));
    }
    let axis = axis_idx as usize;
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

    Tensor::concat(&[part1, part2], axis as isize).map_err(|e| e.to_string())
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

    #[test]
    fn test_roll_accepts_a_scalar_value() {
        let res = process(Payload::scalar_f32(2.0));
        match res {
            Payload::Scalar(out) => {
                assert_eq!(out.as_f32_slice().unwrap(), &[2.0]);
            }
            other => panic!(
                "scalar path produced the wrong payload: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
