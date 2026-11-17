use core_types::numeric::{self, FloatElement, FloatVals};
use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
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

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let args = &prepared.args;
    let mut eps32 = 1e-5f32;
    let mut eps64 = 1e-5f64;
    if let Some(e_str) = args
        .get_named("eps")
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(e) = prepared.args.parse::<f32>(e_str) {
            eps32 = e;
        }
        if let Ok(e) = e_str.parse::<f64>() {
            eps64 = e;
        }
    }

    match payload.into_unwrapped() {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            let shape = tensor.shape.to_vec();
            let result = match numeric::float_vals(&tensor) {
                FloatVals::F32(vals) => rms_norm::<f32>(vals, shape, eps32),
                FloatVals::F64(vals) => rms_norm::<f64>(vals, shape, eps64),
            };
            match result {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'rms_norm\' requires a tensor or scalar value",
        )),
    }
}

/// Normalises every vector along the last axis. F32 is computed in single
/// precision, everything else in F64.
fn rms_norm<T: FloatElement>(
    mut vals: Vec<T>,
    shape: Vec<usize>,
    eps: T,
) -> Result<Tensor, core_types::RString> {
    let feat_dim = shape.last().copied().unwrap_or(0);
    if shape.is_empty() || feat_dim == 0 {
        return Tensor::from_vec_of(vals, shape);
    }
    let n = T::from_i128(feat_dim as i128);
    vals.par_chunks_mut(feat_dim).for_each(|feat| {
        let mean_sq: T = feat.iter().map(|&x| x * x).sum::<T>() / n;
        let inv_rms = T::one() / (mean_sq + eps).sqrt();
        for x in feat {
            *x = *x * inv_rms;
        }
    });
    Tensor::from_vec_of(vals, shape)
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
    fn test_rms_norm_action() {
        let tensor = Tensor::from_f32_shape(&[3.0, 4.0], vec![1, 2]).unwrap();
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            let rms = 12.5f32.sqrt();
            assert!((slice[0] - (3.0 / rms)).abs() < 1e-3);
            assert!((slice[1] - (4.0 / rms)).abs() < 1e-3);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_rms_norm_accepts_a_scalar_value() {
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
