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

    let mut base = "e";
    let mut eps = 1e-8f32;

    if let Some(args) = &args_opt {
        if let Some(b) = args
            .get_named("base")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            base = b;
        }
        if let Some(e_str) = args.get_named("eps") {
            if let Ok(e) = prepared.args.parse::<f32>(e_str) {
                eps = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) | Payload::Scalar(mut tensor)
            if tensor.dtype == TensorDType::F32 =>
        {
            let slice = tensor.as_f32_slice_mut();
            match base {
                "2" => slice
                    .par_iter_mut()
                    .for_each(|x| *x = (*x + eps).max(eps).log2()),
                "10" => slice
                    .par_iter_mut()
                    .for_each(|x| *x = (*x + eps).max(eps).log10()),
                _ => slice
                    .par_iter_mut()
                    .for_each(|x| *x = (*x + eps).max(eps).ln()),
            }
            Payload::from_tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'log\' requires a tensor or scalar value",
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
