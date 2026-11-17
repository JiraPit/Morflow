use core_types::numeric::{self, Element, FloatElement, IntElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{match_dtype, DataType, Payload, Shape, ShapeResult, Tensor, TensorDType};
use rayon::prelude::*;

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}
fn shape_impl(input: Shape, _args: PreparedArgs) -> Shape {
    input
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args).into()
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

/// Clips like NumPy's `clip`: `min(max(x, lo), hi)`. NaN in the input or in
/// a bound gives NaN.
#[inline]
fn clip<T: FloatElement>(x: T, lo: T, hi: T) -> T {
    if x.is_nan() || lo.is_nan() || hi.is_nan() {
        return T::nan();
    }
    let y = if x < lo { lo } else { x };
    if y > hi {
        hi
    } else {
        y
    }
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let args = &prepared.args;
    let min_str = args
        .get_named("min")
        .or_else(|| args.positional.first().map(|s| s.as_str()));
    let max_str = args
        .get_named("max")
        .or_else(|| args.positional.get(1).map(|s| s.as_str()));

    match payload.into_unwrapped() {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match clamp_tensor(tensor, min_str, max_str, &prepared) {
                Ok(out) => Payload::from_tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'clamp\' requires a tensor or scalar value",
        )),
    }
}

fn clamp_tensor(
    mut tensor: Tensor,
    min_str: Option<&str>,
    max_str: Option<&str>,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    match tensor.dtype {
        TensorDType::F32 => {
            let mut min_val = f32::NEG_INFINITY;
            let mut max_val = f32::INFINITY;
            if let Some(m) = min_str.and_then(|s| prepared.args.parse::<f32>(s).ok()) {
                min_val = m;
            }
            if let Some(m) = max_str.and_then(|s| prepared.args.parse::<f32>(s).ok()) {
                max_val = m;
            }
            let slice = tensor.as_f32_slice_mut();
            slice
                .par_iter_mut()
                .for_each(|x| *x = clip(*x, min_val, max_val));
            Ok(tensor)
        }
        TensorDType::F64 => {
            let min_val = min_str
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(f64::NEG_INFINITY);
            let max_val = max_str
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(f64::INFINITY);
            tensor
                .map_same::<f64>(|x| clip(x, min_val, max_val))
                .map_err(|e| e.to_string())
        }
        dtype => match_dtype!(dtype,
            int I => {
                let min_val: I = match min_str {
                    Some(text) => numeric::int_arg::<I>("clamp", "min", text)?,
                    None => I::from_i128(I::MIN_I128),
                };
                let max_val: I = match max_str {
                    Some(text) => numeric::int_arg::<I>("clamp", "max", text)?,
                    None => I::from_i128(I::MAX_I128),
                };
                tensor
                    .map_same::<I>(|x| x.max(min_val).min(max_val))
                    .map_err(|e| e.to_string())
            },
            float F => unreachable!("float dtypes are handled above"),
        ),
    }
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
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_clamp_action() {
        let tensor = Tensor::from_f32_slice(&[-2.0, 0.5, 3.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("min"), RString::from("0.0")));
        named.push(Tuple2(RString::from("max"), RString::from("1.0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 0.5, 1.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_clamp_accepts_a_scalar_value() {
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
}
