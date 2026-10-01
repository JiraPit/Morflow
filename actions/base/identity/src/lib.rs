use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload};
use core_types::{Shape, ShapeResult};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Any
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Any
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, _args: A) -> ShapeResult {
    let _args = _args.into();
    ShapeResult::Ok(input)
}

pub fn get_output_value_shape(
    input: core_types::ValueShape,
    _: PreparedArgs,
) -> core_types::ValueShapeResult {
    core_types::ValueShapeResult::Ok(input)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, _prepared: core_types::PreparedData) -> Payload {
    payload.into_unwrapped()
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
        Some(get_output_value_shape),
    )
}
