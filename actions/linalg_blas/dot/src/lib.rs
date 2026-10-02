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
    core_types::composite_contract::dot(input, args)
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
                _ => return Payload::Error("dot expects 2 tensor inputs in composite".into()),
            };
            match compute_dot(t1, t2) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => match compute_dot(&t, &t) {
            Ok(out) => Payload::Tensor(out),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_dot(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    morflow_openblas::dot(a, b)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    if let Err(reason) = morflow_openblas::dimension_limits(&input, "dot") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "dot",
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
        core_types::shapecheck::execute("dot", shapecheck, super::process, payload)
    }
    use core_types::{RVec, Tensor};

    #[test]
    fn test_dot_action() {
        let t1 = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);
        let t2 = Tensor::from_f32_slice(&[4.0, 5.0, 6.0]);

        let mut items = RVec::new();
        items.push(Payload::Tensor(t1));
        items.push(Payload::Tensor(t2));

        let res = process(Payload::Composite(items));
        if let Payload::Tensor(out) = res {
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert_eq!(out.as_f32_slice().unwrap(), &[32.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[cfg(test)]
#[path = "../../../../backends/openblas/tests/action_contract.rs"]
mod backend_contract;
