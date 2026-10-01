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
    core_types::composite_contract::concat(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut axis = 0isize;
    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = prepared.args.parse::<isize>(ax_str) {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Composite(items) => {
            let mut tensors = Vec::new();
            for item in items {
                match item {
                    Payload::Tensor(t) => tensors.push(t),
                    _ => {
                        return Payload::Error(
                            "All items in composite payload must be tensors for concat".into(),
                        )
                    }
                }
            }
            match Tensor::concat(&tensors, axis) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e),
            }
        }
        Payload::Tensor(t) => Payload::Tensor(t),
        _ => Payload::Error(core_types::RString::from(
            "Action \'concat\' requires Payload::Composite or Payload::Tensor",
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
        None,
        Some(get_output_value_shape),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::{ActionArgs, RBox, RString, RVec, Tensor, Tuple2};

    #[test]
    fn test_concat_action() {
        let t1 = Tensor::from_f32_shape(&[1.0, 2.0], vec![1, 2]).unwrap();
        let t2 = Tensor::from_f32_shape(&[3.0, 4.0], vec![1, 2]).unwrap();

        let mut items = RVec::new();
        items.push(Payload::Tensor(t1));
        items.push(Payload::Tensor(t2));

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Composite(items)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
