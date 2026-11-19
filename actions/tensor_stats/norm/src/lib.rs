use core_types::numeric::{self, Element, FloatElement, IntElement};
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
        .or_else(|| args.positional.get(1).map(|s| s.as_str()));
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
        .or_else(|| args.positional.get(1).map(|s| s.as_str()))
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

/// The requested norm: `p` in both precisions, or the infinity norm.
#[derive(Clone, Copy)]
struct Order {
    p32: f32,
    p64: f64,
    is_inf: bool,
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut order = Order {
        p32: 2.0,
        p64: 2.0,
        is_inf: false,
    };
    let axis = prepared.unsigned("axis").map(|value| value as usize);

    if let Some(args) = &args_opt {
        if let Some(p_str) = args
            .get_named("p")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if p_str == "inf" || p_str == "infinity" {
                order.is_inf = true;
            } else {
                if let Ok(p) = prepared.args.parse::<f32>(p_str) {
                    order.p32 = p;
                }
                if let Ok(p) = p_str.parse::<f64>() {
                    order.p64 = p;
                }
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_norm(
                &tensor,
                order,
                axis,
                prepared
                    .output_dims()
                    .expect("shapecheck prepared dimensions"),
            ) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'norm\' requires a tensor or scalar value".into()),
    }
}

/// Floats keep their dtype; integers give an F64 that is exact (rounded
/// once) for the infinity norm and for positive integer orders.
fn reduce_norm(
    tensor: &Tensor,
    order: Order,
    axis: Option<usize>,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    match_dtype!(tensor.dtype,
        int I => {
            let vals = tensor.to_vec_of::<I>().unwrap_or_default();
            let p = order.p64;
            let exact_order = ((1.0..=64.0).contains(&p) && p.fract() == 0.0).then_some(p as u32);
            let lane_norm = |lane: &[I]| -> f64 {
                if order.is_inf {
                    return lane.iter().map(|v| v.unsigned_abs_u128()).max().unwrap_or(0) as f64;
                }
                match exact_order {
                    Some(p) => numeric::int_p_norm(lane.iter().map(|v| v.to_i128()), p),
                    None => {
                        let sum_p: f64 = lane.iter().map(|v| v.to_f64().abs().powf(p)).sum();
                        sum_p.powf(1.0 / p)
                    }
                }
            };
            let out = match axis {
                Some(ax) => numeric::reduce_axis(&vals, &tensor.shape, ax, lane_norm),
                None => vec![lane_norm(&vals)],
            };
            Tensor::from_vec_of(out, out_shape).map_err(|e| e.to_string())
        },
        float F => {
            let vals = tensor.to_vec_of::<F>().unwrap_or_default();
            let p = F::lit(order.p32, order.p64);
            norm_float(vals, &tensor.shape, p, order.is_inf, axis, out_shape)
        },
    )
}

fn norm_float<T: FloatElement>(
    vals: Vec<T>,
    shape: &[usize],
    p: T,
    is_inf: bool,
    axis: Option<usize>,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    let out = match axis {
        Some(ax) => {
            let inv_p = T::one() / p;
            numeric::reduce_axis(&vals, shape, ax, |lane: &[T]| {
                if is_inf {
                    let mut max_abs = T::zero();
                    for &v in lane {
                        max_abs = max_abs.nan_max(v.abs());
                    }
                    max_abs
                } else {
                    let mut sum_p = T::zero();
                    for &v in lane {
                        sum_p = sum_p + v.abs().powf(p);
                    }
                    sum_p.powf(inv_p)
                }
            })
        }
        None => {
            let norm_val = if is_inf {
                vals.par_iter()
                    .map(|&x| x.abs())
                    .reduce(T::zero, |a, b| a.nan_max(b))
            } else {
                let sum_p: T = vals.par_iter().map(|&x| x.abs().powf(p)).sum();
                sum_p.powf(T::one() / p)
            };
            vec![norm_val]
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
    core_types::shapecheck::axis_plan(result, rank, 1, None, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
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
    let payload = Payload::scalar_f32(2.0);
    let core_types::ShapeCheckResult::Ready { prepared, .. } = shapecheck(
        core_types::InputDescriptor::from_payload(&payload),
        core_types::ActionArgs::default(),
    ) else {
        panic!("expected ready")
    };
    let res = crate::process(payload, prepared);
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
