use core_types::numeric::{self, BinaryOp, Element, FloatElement, IntElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{
    match_dtype, DataType, Payload, Tensor, TensorDType, ValueShape, ValueShapeResult,
};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar | DataType::Composite
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

/// The scalar argument, if one was given.
fn value_arg(args: &core_types::ActionArgs) -> Option<&str> {
    args.get_named("exponent")
        .or_else(|| args.get_named("power"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
}

/// A Tensor or Scalar with a scalar argument keeps its shape; a Composite
/// `[a, b]` broadcasts like NumPy.
pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: ValueShape,
    args: A,
) -> ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::elementwise(input, value_arg(&args).is_some())
}

/// `a ** b` between two tensors.
struct Op;

impl BinaryOp for Op {
    fn float<T: FloatElement>(&self, a: T, b: T) -> T {
        a.powf(b)
    }
    fn int<I: IntElement>(&self, a: I, b: I) -> I {
        a.wrapping_pow_u64(b.to_i128() as u64)
    }
    fn check_int<I: IntElement>(&self, rhs: &[I]) -> Result<(), String> {
        if rhs.iter().any(|e| e.to_i128() < 0) {
            return Err("Action 'pow': integers cannot be raised to a negative exponent".into());
        }
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let inner_payload = payload.into_unwrapped();
    let value = value_arg(&prepared.args);

    match inner_payload {
        Payload::Composite(items) => {
            if value.is_some() {
                return Payload::Error(
                    "Action 'pow' cannot combine a Composite input [a, b] with a scalar argument"
                        .into(),
                );
            }
            match numeric::apply_binary(&items, "pow", &Op) {
                Ok(out) => out,
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match with_scalar(tensor, value, &prepared) {
                Ok(out) => Payload::from_tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'pow\' requires a tensor or scalar value, or a Composite [a, b]",
        )),
    }
}

/// The scalar-argument form: `x ** value` for every element.
fn with_scalar(
    mut tensor: Tensor,
    value: Option<&str>,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    match tensor.dtype {
        TensorDType::F32 => {
            let mut exponent = 1.0f32;
            if let Some(v_str) = value {
                if let Ok(v) = prepared.args.parse::<f32>(v_str) {
                    exponent = v;
                }
            }
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x = x.powf(exponent));
            Ok(tensor)
        }
        TensorDType::F64 => {
            let exponent = value.and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
            tensor
                .map_same::<f64>(|x| x.powf(exponent))
                .map_err(|e| e.to_string())
        }
        dtype => match_dtype!(dtype,
            int I => {
                let exponent: u64 = match value {
                    Some(text) => {
                        if numeric::parse_whole(text).is_some_and(|e| e < 0) {
                            return Err(format!(
                                "Action 'pow': an integer tensor cannot be raised to a negative exponent ({text})"
                            ));
                        }
                        numeric::int_arg::<I>("pow", "exponent", text)?.to_i128() as u64
                    }
                    None => 1,
                };
                tensor
                    .map_same::<I>(|x| x.wrapping_pow_u64(exponent))
                    .map_err(|e| e.to_string())
            },
            float F => unreachable!("float dtypes are handled above"),
        ),
    }
}

#[no_mangle]
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
        None,
        Some(get_output_value_shape),
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
    fn test_pow_action() {
        let tensor = Tensor::from_f32_slice(&[2.0, 3.0, 4.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("exponent"), RString::from("2.0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[4.0, 9.0, 16.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_pow_accepts_a_scalar_value() {
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
