use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite | DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: core_types::ValueShape,
    args: A,
) -> core_types::ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::matmul(input, args)
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
        Payload::Composite(items) if items.len() == 2 => {
            let (t1, t2) = match (&items[0], &items[1]) {
                (Payload::Tensor(a), Payload::Tensor(b)) => (a, b),
                _ => return Payload::Error("matmul expects 2 tensor inputs in composite".into()),
            };
            match compute_matmul(t1, t2, &prepared) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => {
            // Self-multiplication A * A
            match compute_matmul(&t, &t, &prepared) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        other => other,
    }
}

fn compute_matmul(
    a: &Tensor,
    b: &Tensor,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    morflow_openblas::matmul(a, b, prepared)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    if let Err(reason) = morflow_openblas::dimension_limits(&input, "matmul") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "matmul",
        get_input_type(),
        get_output_type(),
        None,
        Some(get_output_value_shape),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute("matmul", shapecheck, super::process, payload)
    }
    use core_types::{RVec, Tensor};

    #[test]
    fn test_matmul_action() {
        // [2, 3] x [3, 2] -> [2, 2]
        let a = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_f32_shape(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0], vec![3, 2]).unwrap();

        let mut items = RVec::new();
        items.push(Payload::Tensor(a));
        items.push(Payload::Tensor(b));

        let res = process(Payload::Composite(items));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            // [1*7+2*9+3*2, 1*8+2*1+3*3] = [7+18+6, 8+2+9] = [31, 19]
            // [4*7+5*9+6*2, 4*8+5*1+6*3] = [28+45+12, 32+5+18] = [85, 55]
            assert_eq!(out.as_f32_slice().unwrap(), &[31.0, 19.0, 85.0, 55.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[cfg(test)]
#[path = "../../../../backends/openblas/tests/action_contract.rs"]
mod backend_contract;
