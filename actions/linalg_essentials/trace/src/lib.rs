use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if input.rank() < 2 {
            return Err("trace requires rank at least 2".into());
        }
        let rank = input.rank();
        contract::shape(input.dims()[..rank - 2].iter().copied())
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, _) = (payload.into_unwrapped(), Some(&prepared.args));

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_trace(&tensor) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error("Action 'trace' requires a tensor".into()),
    }
}

fn compute_trace(tensor: &Tensor) -> Result<Tensor, String> {
    let r = tensor.rank();

    let h = tensor.shape[r - 2];
    let w = tensor.shape[r - 1];
    let diag_len = h.min(w);

    let batch_size: usize = tensor.shape[0..r - 2].iter().product();
    let mat_size = h * w;
    let vals = tensor.to_vec_f32();

    let mut out_traces = vec![0.0f32; batch_size];

    out_traces.par_iter_mut().enumerate().for_each(|(b, tr)| {
        let offset = b * mat_size;
        let mut sum = 0.0f32;
        for i in 0..diag_len {
            sum += vals[offset + i * w + i];
        }
        *tr = sum;
    });

    let out_shape = if r > 2 {
        tensor.shape[0..r - 2].to_vec()
    } else {
        Vec::new()
    };

    Tensor::from_f32_vec(out_traces, out_shape).map_err(|e| e.to_string())
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
    fn test_trace_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(tensor));
        if let Payload::Scalar(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[5.0]);
        } else {
            panic!("Expected Scalar output");
        }
    }
    #[test]
    fn test_trace_returns_scalar_for_tensor_input() {
        let tensor = Tensor::from_f32_shape(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(tensor));
        match res {
            Payload::Scalar(out) => {
                assert!(out.as_f32_slice().is_some());
            }
            other => panic!(
                "action did not return a scalar: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
