use crate::interface;
mod cholesky;
mod concat;
mod det;
mod dot;
mod inv;
mod matmul;
mod outer;
mod qr;
mod repeat;
mod roll;

pub fn dispatch(
    operation: u32,
    payload: core_types::Payload,
    prepared: core_types::PreparedData,
) -> core_types::Payload {
    match operation {
        interface::operation!(cholesky) => cholesky::process_impl(payload, prepared),
        interface::operation!(det) => det::process_impl(payload, prepared),
        interface::operation!(dot) => dot::process_impl(payload, prepared),
        interface::operation!(inv) => inv::process_impl(payload, prepared),
        interface::operation!(matmul) => matmul::process_impl(payload, prepared),
        interface::operation!(outer) => outer::process_impl(payload, prepared),
        interface::operation!(qr) => qr::process_impl(payload, prepared),
        interface::operation!(concat) => concat::process_impl(payload, prepared),
        interface::operation!(repeat) => repeat::process_impl(payload, prepared),
        interface::operation!(roll) => roll::process_impl(payload, prepared),
        _ => core_types::Payload::Error(format!("Unknown OpenBLAS operation: {operation}").into()),
    }
}
