use core_types::numeric::{self, FloatElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{match_dtype, DataType, Payload, Shape, ShapeResult, Tensor};
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
    let inner_payload = payload.into_unwrapped();

    let axis = prepared.unsigned("axis").map(|value| value as usize);

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_min(
                &tensor,
                axis,
                prepared
                    .output_dims()
                    .expect("shapecheck prepared dimensions"),
            ) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'min\' requires a tensor or scalar value".into()),
    }
}

/// The minimum keeps the input dtype; NaN propagates like NumPy.
fn reduce_min(
    tensor: &Tensor,
    axis: Option<usize>,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    match_dtype!(tensor.dtype,
        int I => {
            let vals = tensor.to_vec_of::<I>().unwrap_or_default();
            let out = match axis {
                Some(ax) => numeric::reduce_axis(&vals, &tensor.shape, ax, |lane: &[I]| {
                    lane.iter().copied().reduce(|a, b| a.min(b)).unwrap_or_default()
                }),
                None => {
                    let Some(m) = vals.par_iter().copied().reduce_with(|a, b| a.min(b)) else {
                        return Err("Action \'min\' cannot reduce an empty integer tensor: it has no identity value".into());
                    };
                    vec![m]
                }
            };
            Tensor::from_vec_of(out, out_shape).map_err(|e| e.to_string())
        },
        float F => {
            let vals = tensor.to_vec_of::<F>().unwrap_or_default();
            let out = match axis {
                Some(ax) => numeric::reduce_axis(&vals, &tensor.shape, ax, |lane: &[F]| {
                    let mut m = F::INFINITY;
                    for &v in lane {
                        m = m.nan_min(v);
                    }
                    m
                }),
                None => vec![vals.par_iter().copied().reduce(|| F::INFINITY, |a, b| a.nan_min(b))],
            };
            Tensor::from_vec_of(out, out_shape).map_err(|e| e.to_string())
        },
    )
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
    fn test_min_action() {
        let tensor = Tensor::from_f32_slice(&[5.0, 1.0, 10.0, -2.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Scalar(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[-2.0]);
        } else {
            panic!("Expected Scalar output");
        }
    }
}

#[test]
fn test_min_accepts_a_scalar_value() {
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
