use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    let axis_arg = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()));
    if axis_arg.is_some_and(|value| value.starts_with('$')) {
        let keep = args.get_named("keepdim");
        if keep.is_some_and(|value| value.starts_with('$')) {
            return ShapeResult::Unknown;
        }
        let keep = keep.is_some_and(|value| value == "true" || value == "1");
        let rank = if keep {
            input.rank()
        } else {
            input.rank().saturating_sub(1)
        };
        return ShapeResult::Ok(Shape::unknown(rank));
    }
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
        if args.parse::<isize>(value).is_err() {
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
        if let Ok(ax) = args.parse::<isize>(ax_str) {
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
    if input.dims()[ax] == 0 {
        return ShapeResult::Invalid(
            "Cannot reduce this operation along an empty dimension".into(),
        );
    }

    let mut out: Vec<core_types::Dimension> = Vec::new();
    for (i, &d) in input.dims().iter().enumerate() {
        if i == ax {
            if keepdim {
                out.push(1.into());
            }
        } else {
            out.push(d);
        }
    }
    ShapeResult::Ok(Shape::new(out))
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let axis = prepared.unsigned("axis").map(|value| value as usize);
    let mut unbiased = true;

    if let Some(args) = &args_opt {
        if let Some(unb_str) = args.get_named("unbiased") {
            unbiased = unb_str != "false" && unb_str != "0";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_var(
                &tensor,
                axis,
                unbiased,
                prepared
                    .output_dims()
                    .expect("shapecheck prepared dimensions"),
            ) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'var\' requires a tensor or scalar value".into()),
    }
}

fn reduce_var(
    tensor: &Tensor,
    axis: Option<usize>,
    unbiased: bool,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();

    if let Some(ax) = axis {
        let outer_size: usize = tensor.shape[0..ax].iter().product();
        let axis_len = tensor.shape[ax];

        let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

        let out_len = outer_size * inner_size;
        if out_len == 0 {
            return Tensor::from_f32_vec(Vec::new(), out_shape).map_err(|e| e.to_string());
        }
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

        Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
    } else {
        let n = vals.len();
        if n == 0 {
            let shape = out_shape;
            return Tensor::from_f32_vec(vec![0.0], shape).map_err(|e| e.to_string());
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

        let shape = out_shape;
        Tensor::from_f32_vec(vec![var], shape).map_err(|e| e.to_string())
    }
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    let rank = input.value.shape().map(Shape::rank);
    let result = core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    );
    core_types::shapecheck::axis_plan(result, rank, 0, None, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
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
