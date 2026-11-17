use core_types::numeric::{self, Element, FloatElement, IntMoments};
use core_types::shapecheck::PreparedArgs;
use core_types::{match_dtype, DataType, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[cfg_attr(not(morflow_pack), no_mangle)]
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

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let inner_payload = payload.into_unwrapped();

    let axis = prepared.unsigned("axis").map(|value| value as usize);
    let mut unbiased = true;
    if let Some(unb_str) = prepared.args.get_named("unbiased") {
        unbiased = unb_str != "false" && unb_str != "0";
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

/// Floats keep their dtype; integers give an exactly computed F64.
fn reduce_var(
    tensor: &Tensor,
    axis: Option<usize>,
    unbiased: bool,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    match_dtype!(tensor.dtype,
        int I => {
            let vals = tensor.to_vec_of::<I>().unwrap_or_default();
            let moments_of = |lane: &[I]| {
                let mut moments = IntMoments::default();
                lane.iter().for_each(|v| moments.push(v.to_i128()));
                moments.var(unbiased)
            };
            let out = match axis {
                Some(ax) => numeric::reduce_axis(&vals, &tensor.shape, ax, moments_of),
                None => vec![moments_of(&vals)],
            };
            Tensor::from_vec_of(out, out_shape).map_err(|e| e.to_string())
        },
        float F => var_float::<F>(tensor.to_vec_of::<F>().unwrap_or_default(), &tensor.shape, axis, unbiased, out_shape),
    )
}

fn var_float<T: FloatElement>(
    vals: Vec<T>,
    shape: &[usize],
    axis: Option<usize>,
    unbiased: bool,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    let count = |n: usize| T::from_i128(n as i128);
    let out = match axis {
        Some(ax) => {
            let axis_len = shape[ax];
            let divisor = if unbiased && axis_len > 1 {
                count(axis_len - 1)
            } else {
                count(axis_len)
            };
            numeric::reduce_axis(&vals, shape, ax, |lane: &[T]| {
                let mut sum = T::zero();
                for &v in lane {
                    sum = sum + v;
                }
                let mean = sum / count(axis_len);

                let mut sum_sq_diff = T::zero();
                for &v in lane {
                    let diff = v - mean;
                    sum_sq_diff = sum_sq_diff + diff * diff;
                }
                sum_sq_diff / divisor
            })
        }
        None => {
            let n = vals.len();
            if n == 0 {
                return Tensor::from_vec_of(vec![T::zero()], out_shape).map_err(|e| e.to_string());
            }
            let sum: T = vals.par_iter().copied().sum();
            let mean = sum / count(n);
            let sum_sq_diff: T = vals.par_iter().map(|&x| (x - mean) * (x - mean)).sum();
            let divisor = if unbiased && n > 1 {
                count(n - 1)
            } else {
                count(n)
            };
            vec![sum_sq_diff / divisor]
        }
    };
    Tensor::from_vec_of(out, out_shape).map_err(|e| e.to_string())
}

#[cfg_attr(not(morflow_pack), no_mangle)]
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
