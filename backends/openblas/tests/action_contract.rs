//! Included by each action's unit tests to verify the actual exported callbacks.
use super::*;
use core_types::{ActionArgs, InputDescriptor, Payload, RString, RVec, ShapeCheckResult, Tensor};
#[test]
fn shapecheck_does_not_load_openblas_and_process_reports_missing_library() {
    const FLAG: &str = "MORFLOW_BLAS_ACTION_CONTRACT_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let result=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","backend_contract::shapecheck_does_not_load_openblas_and_process_reports_missing_library"])
            .env(FLAG,"1").env("MORFLOW_OPENBLAS_LIBRARY","/definitely/missing/openblas.so").output().unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let action = env!("CARGO_PKG_NAME").rsplit('_').next().unwrap();
    if env!("CARGO_PKG_NAME").starts_with("linalg_blas_") {
        let too_large =
            core_types::ValueShape::tensor(core_types::Shape::new([i32::MAX as usize + 1, 1]));
        let sample = matrix_for_descriptor(too_large, action);
        let result = shapecheck(sample, ActionArgs::default());
        assert!(matches!(result,ShapeCheckResult::Invalid {reason} if reason.contains("LP64")));
    }
    let matrix = Tensor::from_f32_shape(&[4., 1., 1., 3.], vec![2, 2]).unwrap();
    let vector = Tensor::from_f32_slice(&[1., 2.]);
    let mut args = ActionArgs::default();
    let payload = match action {
        "matmul" | "concat" => Payload::Composite(RVec::from(vec![
            Payload::Tensor(matrix.clone()),
            Payload::Tensor(matrix),
        ])),
        "dot" | "outer" => Payload::Composite(RVec::from(vec![
            Payload::Tensor(vector.clone()),
            Payload::Tensor(vector),
        ])),
        "repeat" => {
            args.positional.push(RString::from("2,1"));
            Payload::Tensor(matrix)
        }
        _ => Payload::Tensor(matrix),
    };
    let result = shapecheck(InputDescriptor::from_payload(&payload), args);
    let ShapeCheckResult::Ready { prepared, .. } = result else {
        panic!("Concrete shapes must check without OpenBLAS: {result:?}")
    };
    let result = super::process(payload, prepared);
    let Payload::Error(error) = result else {
        panic!("Missing library must fail only during processing")
    };
    assert!(error.contains("MORFLOW_OPENBLAS_LIBRARY"), "{error}");
}

fn matrix_for_descriptor(value: core_types::ValueShape, action: &str) -> InputDescriptor {
    let payload = Payload::Tensor(Tensor::from_f32_shape(&[1.], vec![1, 1]).unwrap());
    let mut input = InputDescriptor::from_payload(&payload);
    input.value = if matches!(action, "matmul" | "dot" | "outer") {
        core_types::ValueShape::composite([value.clone(), value])
    } else {
        value
    };
    input
}
