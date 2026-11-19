use core_types::numeric::{self, BinaryOp, Element, FloatElement, IntElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{
    match_dtype, DataType, Payload, Tensor, TensorDType, ValueShape, ValueShapeResult,
};
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

/// `a + b` between two tensors.
struct Op;

impl BinaryOp for Op {
    fn float<T: FloatElement>(&self, a: T, b: T) -> T {
        a + b
    }
    fn int<I: IntElement>(&self, a: I, b: I) -> I {
        a.wrapping_add(b)
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
                    "Action 'add' cannot combine a Composite input [a, b] with a scalar argument"
                        .into(),
                );
            }
            match numeric::apply_binary(&items, "add", &Op) {
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
            "Action \'add\' requires a tensor or scalar value, or a Composite [a, b]",
        )),
    }
}

/// The scalar-argument form: `x + value` for every element.
fn with_scalar(
    mut tensor: Tensor,
    value: Option<&str>,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    match tensor.dtype {
        TensorDType::F32 => {
            let mut add_val = 0.0f32;
            if let Some(v_str) = value {
                if let Ok(v) = prepared.args.parse::<f32>(v_str) {
                    add_val = v;
                }
            }
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x += add_val);
            Ok(tensor)
        }
        TensorDType::F64 => {
            let add_val = value.and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            tensor
                .map_same::<f64>(|x| x + add_val)
                .map_err(|e| e.to_string())
        }
        dtype => match_dtype!(dtype,
            int I => {
                let add_val: I = match value {
                    Some(text) => numeric::int_arg::<I>("add", "value", text)?,
                    None => I::from_i128(0),
                };
                tensor
                    .map_same::<I>(|x| x.wrapping_add(add_val))
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
    fn test_add_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("value"), RString::from("5.0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[6.0, 7.0, 8.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_add_accepts_a_scalar_value() {
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
