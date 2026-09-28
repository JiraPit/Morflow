use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::RawBytes
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::RawBytes
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    match payload {
        Payload::WithArgs { payload, .. } => payload.as_ref().clone(),
        other => other,
    }
}
