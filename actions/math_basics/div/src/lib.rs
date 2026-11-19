use core_types::numeric::{self, BinaryOp, FloatElement, IntElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Tensor, TensorDType, ValueShape, ValueShapeResult};
use rayon::prelude::*;

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar | DataType::Composite
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

/// The scalar argument, if one was given.
fn value_arg(args: &core_types::ActionArgs) -> Option<&str> {
    args.get_named("value")
        .or_else(|| args.get_named("val"))
        .or_else(|| args.get_named("scalar"))
        .or_else(|| args.get_named("by"))
        .or_else(|| args.get_named("amount"))
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

/// `a / b` between two tensors.
struct Op;

impl BinaryOp for Op {
    const FRACTIONAL: bool = true;
    fn float<T: FloatElement>(&self, a: T, b: T) -> T {
        a / b
    }
    fn int<I: IntElement>(&self, _a: I, _b: I) -> I {
        unreachable!("integer division always produces F64")
    }
}

#[cfg_attr(not(morflow_pack), no_mangle)]
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
                    "Action 'div' cannot combine a Composite input [a, b] with a scalar argument"
                        .into(),
                );
            }
            match numeric::apply_binary(&items, "div", &Op) {
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
            "Action \'div\' requires a tensor or scalar value, or a Composite [a, b]",
        )),
    }
}

/// The scalar-argument form: `x / value` for every element.
fn with_scalar(
    mut tensor: Tensor,
    value: Option<&str>,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    match tensor.dtype {
        TensorDType::F32 => {
            let mut div_val = 1.0f32;
            if let Some(v_str) = value {
                if let Ok(v) = prepared.args.parse::<f32>(v_str) {
                    div_val = v;
                }
            }
            if div_val == 0.0 {
                return Err("Division by zero in div action".into());
            }
            let inv = 1.0 / div_val;
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x *= inv);
            Ok(tensor)
        }
        dtype => {
            // F64 divides in double precision; integers become F64.
            let div_val = value.and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
            if div_val == 0.0 {
                return Err("Division by zero in div action".into());
            }
            if dtype == TensorDType::F64 {
                return tensor
                    .map_same::<f64>(|x| x / div_val)
                    .map_err(|e| e.to_string());
            }
            let mut vals = tensor.to_vec_f64();
            vals.par_iter_mut().for_each(|x| *x /= div_val);
            Tensor::from_vec_of(vals, tensor.shape.to_vec()).map_err(|e| e.to_string())
        }
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
    fn test_div_action() {
        let tensor = Tensor::from_f32_slice(&[10.0, 20.0, 30.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("value"), RString::from("2.0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[5.0, 10.0, 15.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_div_accepts_a_scalar_value() {
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
