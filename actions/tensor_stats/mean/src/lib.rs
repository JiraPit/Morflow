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

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_mean(
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
        _ => Payload::Error("Action \'mean\' requires a tensor or scalar value".into()),
    }
}

/// Floats keep their dtype; integers give an exactly computed F64.
fn reduce_mean(
    tensor: &Tensor,
    axis: Option<usize>,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    match_dtype!(tensor.dtype,
        int I => {
            let vals = tensor.to_vec_of::<I>().unwrap_or_default();
            let out = match axis {
                Some(ax) => numeric::reduce_axis(&vals, &tensor.shape, ax, |lane: &[I]| {
                    if lane.is_empty() {
                        return f64::NAN;
                    }
                    let mut moments = IntMoments::default();
                    lane.iter().for_each(|v| moments.push(v.to_i128()));
                    moments.mean()
                }),
                None => {
                    let mut moments = IntMoments::default();
                    vals.iter().for_each(|v| moments.push(v.to_i128()));
                    vec![moments.mean()]
                }
            };
            Tensor::from_vec_of(out, out_shape).map_err(|e| e.to_string())
        },
        float F => mean_float::<F>(tensor.to_vec_of::<F>().unwrap_or_default(), &tensor.shape, axis, out_shape),
    )
}

fn mean_float<T: FloatElement>(
    vals: Vec<T>,
    shape: &[usize],
    axis: Option<usize>,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    let out = match axis {
        Some(ax) => {
            let inv_len = T::one() / T::from_i128(shape[ax] as i128);
            numeric::reduce_axis(&vals, shape, ax, |lane: &[T]| {
                let mut acc = T::zero();
                for &v in lane {
                    acc = acc + v;
                }
                acc * inv_len
            })
        }
        None => {
            let total: T = vals.par_iter().copied().sum();
            let mean = if !vals.is_empty() {
                total / T::from_i128(vals.len() as i128)
            } else {
                T::zero()
            };
            vec![mean]
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
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_mean_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 3.0, 5.0, 7.0], vec![2, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[2.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[test]
fn test_mean_accepts_a_scalar_value() {
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
