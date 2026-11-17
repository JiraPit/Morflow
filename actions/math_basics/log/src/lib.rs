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

/// `log(x + eps)` in base e, 2 or 10, with the argument floored at `eps`; NaN is kept.
struct Op {
    eps32: f32,
    eps64: f64,
    base: u32,
}

impl FloatUnary for Op {
    #[inline]
    fn apply<T: FloatElement>(&self, x: T) -> T {
        if x.is_nan() {
            return x;
        }
        let eps = T::lit(self.eps32, self.eps64);
        let y = (x + eps).max(eps);
        match self.base {
            2 => y.log2(),
            10 => y.log10(),
            _ => y.ln(),
        }
    }
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let _ = &prepared;
    let args = &prepared.args;
    let base = match args
        .get_named("base")
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        Some("2") => 2,
        Some("10") => 10,
        _ => 0,
    };
    let mut op = Op {
        eps32: 1e-8f32,
        eps64: 1e-8f64,
        base,
    };
    if let Some(e_str) = args.get_named("eps") {
        if let Ok(e) = prepared.args.parse::<f32>(e_str) {
            op.eps32 = e;
        }
        if let Ok(e) = e_str.parse::<f64>() {
            op.eps64 = e;
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
            "Action \'log\' requires a tensor or scalar value",
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
    use core_types::Tensor;

    #[test]
    fn test_log_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, std::f32::consts::E]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!(slice[0].abs() < 1e-4);
            assert!((slice[1] - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_log_accepts_a_scalar_value() {
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
                assert_eq!(out.as_f32_slice().unwrap()[0], (2.0f32 + 1e-8).ln());
            }
            other => panic!(
                "scalar path produced the wrong payload: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
