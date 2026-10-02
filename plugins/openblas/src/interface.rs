// Native interface shared by this plugin and its consuming actions.
pub const PLUGIN_NAME: &str = "openblas";
pub const PLUGIN_VERSION: &str = "^0.1.0";
pub const PROCESS_SYMBOL: &str = "morflow_openblas_process";
pub type NativeProcess =
    extern "C" fn(u32, core_types::Payload, core_types::PreparedData) -> core_types::Payload;

// Expand only the identifier requested by a consumer; no unused constants are emitted.
macro_rules! operation {
    (cholesky) => {
        0
    };
    (det) => {
        1
    };
    (dot) => {
        2
    };
    (inv) => {
        3
    };
    (matmul) => {
        4
    };
    (outer) => {
        5
    };
    (qr) => {
        6
    };
    (concat) => {
        7
    };
    (repeat) => {
        8
    };
    (roll) => {
        9
    };
}
pub(crate) use operation;
