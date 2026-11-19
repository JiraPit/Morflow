use core_types::numeric::{self, FloatElement, FloatUnary};
use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};

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

/// `x` for `x >= 0`, `alpha * x` otherwise.
struct Op {
    alpha32: f32,
    alpha64: f64,
}

impl FloatUnary for Op {
    #[inline]
    fn apply<T: FloatElement>(&self, x: T) -> T {
        if x >= T::zero() {
            x
        } else {
            x * T::lit(self.alpha32, self.alpha64)
        }
    }
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let _ = &prepared;
    let args = &prepared.args;
    let mut op = Op {
        alpha32: 0.01f32,
        alpha64: 0.01f64,
    };
    if let Some(a_str) = args
        .get_named("alpha")
        .or_else(|| args.get_named("negative_slope"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(a) = prepared.args.parse::<f32>(a_str) {
            op.alpha32 = a;
        }
        if let Ok(a) = a_str.parse::<f64>() {
            op.alpha64 = a;
        }
    }

    match payload.into_unwrapped() {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match numeric::apply_float_unary(tensor, &op) {
                Ok(out) => Payload::from_tensor(out),
                Err(e) => Payload::Error(e),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'leaky_relu\' requires a tensor or scalar value",
        )),
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
    fn test_leaky_relu_action() {
        let tensor = Tensor::from_f32_slice(&[-2.0, 0.0, 3.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("alpha"), RString::from("0.1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[-0.2, 0.0, 3.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_leaky_relu_accepts_a_scalar_value() {
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
