use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    // Validate the same selected argument that execution consumes.
    for value in [
        core_types::contract::value(&args, &["axis", "dim"], Some(0)),
        core_types::contract::value(&args, &["keepdim"], None),
    ] {
        if let Err(error) = value {
            return core_types::contract::finish(Err(error));
        }
    }
    if let Some(value) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if value.parse::<isize>().is_err() {
            return ShapeResult::Invalid(format!("Invalid axis argument '{value}'").into());
        }
    }

    let r = input.rank();
    let mut axis: Option<isize> = None;
    let mut keepdim = false;
    if let Some(ax_str) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(ax) = ax_str.parse::<isize>() {
            axis = Some(ax);
        }
    }
    if let Some(kd_str) = args.get_named("keepdim") {
        keepdim = kd_str == "true" || kd_str == "1";
    }

    let Some(raw) = axis else {
        let out = if keepdim {
            vec![1; r.max(1)]
        } else {
            Vec::new()
        };
        return ShapeResult::Ok(Shape::new(out));
    };
    let resolved = if raw < 0 { raw + r as isize } else { raw };
    if !(0..r as isize).contains(&resolved) {
        return ShapeResult::Invalid(core_types::reducer_axis_reason(raw, r).into());
    }
    let ax = resolved as usize;
    let mut out: Vec<usize> = Vec::new();
    for (i, &d) in input.dims().iter().enumerate() {
        if i == ax {
            if keepdim {
                out.push(1);
            }
        } else {
            out.push(d);
        }
    }
    ShapeResult::Ok(Shape::new(out))
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut axis = None;
    let mut unbiased = true;
    let mut keepdim = false;

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
        if let Some(unb_str) = args.get_named("unbiased") {
            unbiased = unb_str != "false" && unb_str != "0";
        }
        if let Some(kd_str) = args.get_named("keepdim") {
            keepdim = kd_str == "true" || kd_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_var(&tensor, axis, unbiased, keepdim) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'var\' requires a tensor or scalar value".into()),
    }
}

fn reduce_var(
    tensor: &Tensor,
    axis: Option<isize>,
    unbiased: bool,
    keepdim: bool,
) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();

    if let Some(raw_ax) = axis {
        let ax = if raw_ax < 0 {
            let pos = raw_ax + r as isize;
            if pos < 0 {
                return Err(format!(
                    "Axis {} out of bounds for tensor of rank {}",
                    raw_ax, r
                ));
            }
            pos as usize
        } else {
            raw_ax as usize
        };

        if ax >= r {
            return Err(format!(
                "Axis {} out of bounds for tensor of rank {}",
                raw_ax, r
            ));
        }

        let outer_size: usize = tensor.shape[0..ax].iter().product();
        let axis_len = tensor.shape[ax];
        if axis_len == 0 {
            return Err("Cannot compute variance along empty dimension".into());
        }
        let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

        let out_len = outer_size * inner_size;
        let mut out_vals = vec![0.0f32; out_len];
        let divisor = if unbiased && axis_len > 1 {
            (axis_len - 1) as f32
        } else {
            axis_len as f32
        };

        out_vals
            .par_chunks_mut(inner_size)
            .enumerate()
            .for_each(|(outer_idx, slice)| {
                for (inner_idx, slot) in slice.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for a in 0..axis_len {
                        let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                        sum += vals[in_idx];
                    }
                    let mean = sum / axis_len as f32;

                    let mut sum_sq_diff = 0.0f32;
                    for a in 0..axis_len {
                        let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                        let diff = vals[in_idx] - mean;
                        sum_sq_diff += diff * diff;
                    }
                    *slot = sum_sq_diff / divisor;
                }
            });

        let mut out_shape = Vec::new();
        for (i, &dim) in tensor.shape.iter().enumerate() {
            if i == ax {
                if keepdim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(dim);
            }
        }

        Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
    } else {
        let n = vals.len();
        if n == 0 {
            return Ok(Tensor::from_f32_slice(&[0.0]));
        }
        let sum: f32 = vals.par_iter().sum();
        let mean = sum / n as f32;
        let sum_sq_diff: f32 = vals.par_iter().map(|&x| (x - mean) * (x - mean)).sum();
        let divisor = if unbiased && n > 1 {
            (n - 1) as f32
        } else {
            n as f32
        };
        let var = sum_sq_diff / divisor;

        let shape = if keepdim {
            vec![1; r.max(1)]
        } else {
            Vec::new()
        };
        Tensor::from_f32_vec(vec![var], shape).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_var_action() {
        let tensor = Tensor::from_f32_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Scalar(out) = res {
            let slice = out.as_f32_slice().unwrap();
            // sample variance = 4.5714
            assert!((slice[0] - 4.5714).abs() < 1e-3);
        } else {
            panic!("Expected Scalar output");
        }
    }
}
