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

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut eps = 1e-5f32;
    if let Some(args) = &args_opt {
        if let Some(e_str) = args
            .get_named("eps")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(e) = prepared.args.parse::<f32>(e_str) {
                eps = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let r = tensor.rank();
            if r == 0 {
                return Payload::from_tensor(tensor);
            }
            let feat_dim = *tensor.shape.last().unwrap();
            if feat_dim == 0 {
                return Payload::from_tensor(tensor);
            }

            let slice = tensor.as_f32_slice_mut();
            slice.par_chunks_mut(feat_dim).for_each(|feat| {
                let mean_sq: f32 = feat.iter().map(|&x| x * x).sum::<f32>() / feat_dim as f32;
                let inv_rms = 1.0 / (mean_sq + eps).sqrt();
                for x in feat {
                    *x *= inv_rms;
                }
            });
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'rms_norm\' requires a tensor or scalar value",
        )),
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
