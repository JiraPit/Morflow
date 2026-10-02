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
        if input.rank() != 2 {
            return Err("cholesky requires a rank-2 matrix".into());
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
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_cholesky(&tensor) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_cholesky(a: &Tensor) -> Result<Tensor, String> {
    morflow_openblas::cholesky(a)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    if let Err(reason) = morflow_openblas::dimension_limits(&input, "cholesky") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "cholesky",
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
        core_types::shapecheck::execute("cholesky", shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_cholesky_action() {
        // A = [4, 12, -16; 12, 37, -43; -16, -43, 98]
        // L = [2, 0, 0; 6, 1, 0; -8, 5, 3]
        let a = Tensor::from_f32_shape(
            &[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
            vec![3, 3],
        )
        .unwrap();

        let res = process(Payload::Tensor(a));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[3, 3]);
            let slice = out.as_f32_slice().unwrap();
            assert_eq!(slice, &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0,]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[cfg(test)]
#[path = "../../../../backends/openblas/tests/action_contract.rs"]
mod backend_contract;
