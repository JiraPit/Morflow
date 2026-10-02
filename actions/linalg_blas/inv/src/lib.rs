use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if input.rank() < 2 {
            return Err("inv requires rank at least 2".into());
        }
        let rank = input.rank();
        let (h, w) = (input.dims()[rank - 2], input.dims()[rank - 1]);
        if !h.is_unknown() && !w.is_unknown() && h != w {
            return Err("Matrix must be square".into());
        }
        Ok(input)
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
    if let Err(error) = morflow_openblas::require() {
        return Payload::Error(error);
    }
    let (inner_payload, _) = (payload.into_unwrapped(), Some(&prepared.args));

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_inv(&tensor) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_inv(tensor: &Tensor) -> Result<Tensor, String> {
    morflow_openblas::inv(tensor)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    if let Err(reason) = morflow_openblas::dimension_limits(&input, "inv") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "inv",
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
        core_types::shapecheck::execute("inv", shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_inv_action() {
        // [4, 7; 2, 6] -> det = 24 - 14 = 10 -> inv = [0.6, -0.7; -0.2, 0.4]
        let mat = Tensor::from_f32_shape(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!((slice[0] - 0.6).abs() < 1e-4);
            assert!((slice[1] - (-0.7)).abs() < 1e-4);
            assert!((slice[2] - (-0.2)).abs() < 1e-4);
            assert!((slice[3] - 0.4).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[cfg(test)]
#[path = "../../../../backends/openblas/tests/action_contract.rs"]
mod backend_contract;
