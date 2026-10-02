// Native interface shared by this plugin and its consuming actions.
// Keep implementation and argument validation in the plugin and actions respectively.
use std::ffi::{c_char, c_void};
pub const PLUGIN_NAME: &str = "opencv-bridge";
pub const PLUGIN_VERSION: &str = "^0.1.0";
pub const PROCESS_SYMBOL: &str = "morflow_opencv_process";
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    pub operation: i32,
    pub mode: i32,
    pub radius: i32,
    pub iterations: i32,
    pub shape: i32,
    pub sigma: f64,
    pub strength: f64,
    pub angle: f64,
    pub fill: f64,
}
pub type NativeProcess = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const Options,
    *mut c_char,
    usize,
) -> i32;

// Expand only the identifier requested by a consumer; no unused constants are emitted.
macro_rules! operation {
    (resize) => {
        0
    };
    (gaussian_blur) => {
        1
    };
    (morphology) => {
        2
    };
    (edge_detect) => {
        3
    };
    (sharpen) => {
        4
    };
    (rotate) => {
        5
    };
}
pub(crate) use operation;
