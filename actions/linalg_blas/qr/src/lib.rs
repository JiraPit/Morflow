use core_types::shapecheck::PreparedArgs;
use core_types::{ActionArgs, Shape, ShapeResult};
use core_types::{DataType, Payload, RVec, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Composite
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, _args: A) -> ShapeResult {
    let _args = _args.into();
    if input.rank() != 2 {
        return ShapeResult::Invalid("qr received an unsupported input rank".into());
    }
    if input.dims()[1].checked_mul(input.dims()[1]).is_none() {
        return ShapeResult::Invalid("QR output matrix element count overflows".into());
    }
    ShapeResult::Unknown
}

pub extern "C" fn get_output_components(
    input: Shape,
    args: ActionArgs,
) -> RVec<core_types::OutputComponent> {
    if matches!(
        get_output_shape(input.clone(), args),
        ShapeResult::Invalid(_)
    ) {
        return RVec::new();
    }
    let n = input.dims()[1];
    vec![
        core_types::OutputComponent {
            kind: DataType::Tensor,
            shape: ShapeResult::Ok(input),
        },
        core_types::OutputComponent {
            kind: DataType::Tensor,
            shape: ShapeResult::Ok(Shape::new([n, n])),
        },
    ]
    .into()
}

pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: core_types::ValueShape,
    args: A,
) -> core_types::ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::qr(input, args)
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
        Payload::Tensor(tensor) => match compute_qr(&tensor) {
            Ok((q, r)) => {
                let mut items = RVec::new();
                items.push(Payload::Tensor(q));
                items.push(Payload::Tensor(r));
                Payload::Composite(items)
            }
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_qr(a: &Tensor) -> Result<(Tensor, Tensor), String> {
    morflow_openblas::qr(a)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    if let Err(reason) = morflow_openblas::dimension_limits(&input, "qr") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "qr",
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        Some(get_output_value_shape),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute("qr", shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_qr_action() {
        let mat = Tensor::from_f32_shape(
            &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            vec![3, 3],
        )
        .unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Composite(items) = res {
            assert_eq!(items.len(), 2);
            if let (Payload::Tensor(q), Payload::Tensor(r)) = (&items[0], &items[1]) {
                assert_eq!(q.shape.as_slice(), &[3, 3]);
                assert_eq!(r.shape.as_slice(), &[3, 3]);
            } else {
                panic!("Expected Tensor Q and R");
            }
        } else {
            panic!("Expected Composite output");
        }
    }
}

#[cfg(test)]
#[path = "../../../../backends/openblas/tests/action_contract.rs"]
mod backend_contract;
