use core_types::{ActionArgs, Shape, ShapeResult};
use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Bytes
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Bytes
}

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, _args: ActionArgs) -> ShapeResult {
    ShapeResult::Ok(input)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    payload.into_unwrapped()
}
