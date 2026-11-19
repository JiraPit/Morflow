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

#[inline]
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x.powi(3))).tanh())
}

#[inline]
fn gelu_tanh_f64(x: f64) -> f64 {
    let k = (2.0f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x.powi(3))).tanh())
}

#[inline]
fn erf(x: f32) -> f32 {
    // Chebyshev approximation for error function (Abramowitz & Stegun 7.1.26)
    let sign = if x < 0.0 { -1.0f64 } else { 1.0f64 };
    let a = (x as f64).abs();
    let p = 0.3275911f64;
    let a1 = 0.254829592f64;
    let a2 = -0.284496736f64;
    let a3 = 1.421413741f64;
    let a4 = -1.453152027f64;
    let a5 = 1.061405429f64;

    let t = 1.0 / (1.0 + p * a);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    (sign * (1.0 - poly * (-a * a).exp())) as f32
}

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

/// GELU, exact (erf) or with the tanh approximation.
struct Op {
    use_tanh: bool,
}

impl FloatUnary for Op {
    #[inline]
    fn apply<T: FloatElement>(&self, x: T) -> T {
        // F32 keeps its established kernels; F64 is computed in double
        // precision with an accurate erf.
        if T::DTYPE == core_types::TensorDType::F32 {
            let x = x.to_f64() as f32;
            let y = if self.use_tanh {
                gelu_tanh(x)
            } else {
                gelu_exact(x)
            };
            T::from_f64(y as f64)
        } else {
            let x = x.to_f64();
            let y = if self.use_tanh {
                gelu_tanh_f64(x)
            } else {
                0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
            };
            T::from_f64(y)
        }
    }
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let _ = &prepared;
    let args = &prepared.args;
    let use_tanh = args
        .get_named("approximate")
        .or_else(|| args.positional.first().map(|s| s.as_str()))
        == Some("tanh");
    let op = Op { use_tanh };

    match payload.into_unwrapped() {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match numeric::apply_float_unary(tensor, &op) {
                Ok(out) => Payload::from_tensor(out),
                Err(e) => Payload::Error(e),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'gelu\' requires a tensor or scalar value",
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
    fn test_gelu_action() {
        let tensor = Tensor::from_f32_slice(&[0.0, 1.0, -1.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!(slice[0].abs() < 1e-4);
            assert!((slice[1] - 0.8413).abs() < 1e-3);
            assert!((slice[2] - (-0.1587)).abs() < 1e-3);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_gelu_accepts_a_scalar_value() {
        let payload = Payload::scalar_f32(2.0);
        let core_types::ShapeCheckResult::Ready { prepared, .. } = shapecheck(
            core_types::InputDescriptor::from_payload(&payload),
            core_types::ActionArgs::default(),
        ) else {
            panic!("expected ready")
        };
        let res = crate::process(payload, prepared);
        match res {
            Payload::Scalar(_) => {}
            other => panic!(
                "scalar path produced the wrong payload: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
