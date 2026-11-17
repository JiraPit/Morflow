use core_types::numeric::{self, Element, IntElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{match_dtype, DataType, Payload, Shape, ShapeResult, Tensor};

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg, axis};
    let rank = input.rank();
    let result = contract::finish((|| {
        let dim = arg::<isize>(&args, &["axis", "dim"], Some(0), Some(0))?.unwrap();
        if input.rank() > 0 {
            axis(dim, input.rank(), false)?;
        }
        Ok(input)
    })());
    match result {
        ShapeResult::Unknown => ShapeResult::Ok(Shape::unknown(rank)),
        other => other,
    }
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut axis = 0isize;

    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = prepared.args.parse::<isize>(ax_str) {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_cumsum(&tensor, axis) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error("Action \'cumsum\' requires a tensor or scalar value".into()),
    }
}

/// Floats keep their dtype; signed integers accumulate into I64 and unsigned
/// ones into U64, wrapping on overflow like NumPy.
fn compute_cumsum(tensor: &Tensor, raw_ax: isize) -> Result<Tensor, String> {
    let r = tensor.rank();
    let axis = if r == 0 {
        0
    } else if raw_ax < 0 {
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

    if r > 0 && axis >= r {
        return Err(format!(
            "Axis {} out of bounds for tensor of rank {}",
            raw_ax, r
        ));
    }

    match_dtype!(tensor.dtype,
        int I => {
            let vals = tensor.to_vec_of::<I>().unwrap_or_default();
            if I::SIGNED {
                scan::<I, i64>(&vals, &tensor.shape, axis, |v| v.to_i128() as i64, |a, b| a.wrapping_add(b))
            } else {
                scan::<I, u64>(&vals, &tensor.shape, axis, |v| v.to_i128() as u64, |a, b| a.wrapping_add(b))
            }
        },
        float F => {
            let vals = tensor.to_vec_of::<F>().unwrap_or_default();
            scan::<F, F>(&vals, &tensor.shape, axis, |v| v, |a, b| a + b)
        },
    )
}

fn scan<T, A>(
    vals: &[T],
    shape: &[usize],
    axis: usize,
    convert: impl Fn(T) -> A + Sync + Send,
    add: impl Fn(A, A) -> A + Sync + Send,
) -> Result<Tensor, String>
where
    T: Copy + Send + Sync,
    A: Element,
{
    if shape.is_empty() {
        let out: Vec<A> = vals.iter().map(|&v| convert(v)).collect();
        return Tensor::from_vec_of(out, vec![]).map_err(|e| e.to_string());
    }
    let out = numeric::map_axis(vals, shape, axis, |lane: &[T], out: &mut [A]| {
        let mut acc = A::default();
        for (a, &v) in lane.iter().enumerate() {
            acc = add(acc, convert(v));
            out[a] = acc;
        }
    });
    Tensor::from_vec_of(out, shape.to_vec()).map_err(|e| e.to_string())
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_cumsum_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[test]
fn test_cumsum_accepts_a_scalar_value() {
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
