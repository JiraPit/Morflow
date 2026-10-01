use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
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

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut use_tanh = false;
    if let Some(args) = &args_opt {
        if let Some(approx) = args
            .get_named("approximate")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if approx == "tanh" {
                use_tanh = true;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let slice = tensor.as_f32_slice_mut();
            if use_tanh {
                slice.par_iter_mut().for_each(|x| *x = gelu_tanh(*x));
            } else {
                slice.par_iter_mut().for_each(|x| *x = gelu_exact(*x));
            }
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'gelu\' requires a tensor or scalar value",
        )),
    }
}

#[no_mangle]
pub extern "C" fn get_action_abi_version() -> u32 {
    core_types::shapecheck::ACTION_ABI_VERSION
}
#[no_mangle]
pub extern "C" fn get_action_abi_layout() -> *const core_types::abi_stable::type_layout::TypeLayout
{
    <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT
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
