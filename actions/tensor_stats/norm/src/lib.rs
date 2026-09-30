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
        core_types::contract::value(&args, &["axis", "dim"], Some(1)),
        core_types::contract::value(&args, &["keepdim"], None),
    ] {
        if let Err(error) = value {
            return core_types::contract::finish(Err(error));
        }
    }
    if let Some(value) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.get(1).map(|s| s.as_str()))
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
        .or_else(|| args.positional.get(1).map(|s| s.as_str()))
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

    let mut p_val = 2.0f32;
    let mut is_inf = false;
    let mut axis = None;
    let mut keepdim = false;

    if let Some(args) = &args_opt {
        if let Some(p_str) = args
            .get_named("p")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if p_str == "inf" || p_str == "infinity" {
                is_inf = true;
            } else if let Ok(p) = p_str.parse::<f32>() {
                p_val = p;
            }
        }
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<isize>() {
                axis = Some(ax);
            }
        }
        if let Some(kd_str) = args.get_named("keepdim") {
            keepdim = kd_str == "true" || kd_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_norm(&tensor, p_val, is_inf, axis, keepdim) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'norm\' requires a tensor or scalar value".into()),
    }
}

fn reduce_norm(
    tensor: &Tensor,
    p: f32,
    is_inf: bool,
    axis: Option<isize>,
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
            return Err("Cannot compute norm along empty dimension".into());
        }
        let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

        let out_len = outer_size * inner_size;
        let mut out_vals = vec![0.0f32; out_len];
        let inv_p = 1.0 / p;

        out_vals
            .par_chunks_mut(inner_size)
            .enumerate()
            .for_each(|(outer_idx, slice)| {
                for (inner_idx, slot) in slice.iter_mut().enumerate() {
                    if is_inf {
                        let mut max_abs = 0.0f32;
                        for a in 0..axis_len {
                            let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                            max_abs = max_abs.max(vals[in_idx].abs());
                        }
                        *slot = max_abs;
                    } else {
                        let mut sum_p = 0.0f32;
                        for a in 0..axis_len {
                            let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                            sum_p += vals[in_idx].abs().powf(p);
                        }
                        *slot = sum_p.powf(inv_p);
                    }
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
        let norm_val = if is_inf {
            vals.par_iter()
                .map(|&x| x.abs())
                .reduce(|| 0.0f32, f32::max)
        } else {
            let sum_p: f32 = vals.par_iter().map(|&x| x.abs().powf(p)).sum();
            sum_p.powf(1.0 / p)
        };

        let shape = if keepdim {
            vec![1; r.max(1)]
        } else {
            Vec::new()
        };
        Tensor::from_f32_vec(vec![norm_val], shape).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_norm_action() {
        let tensor = Tensor::from_f32_slice(&[3.0, 4.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Scalar(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!((slice[0] - 5.0).abs() < 1e-4);
        } else {
            panic!("Expected Scalar output");
        }
    }
}

#[test]
fn test_norm_accepts_a_scalar_value() {
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
