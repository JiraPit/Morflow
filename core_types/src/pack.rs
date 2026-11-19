//! Pack libraries: every action of one pack linked into a single native
//! library, for apps that bundle actions instead of loading them from the
//! action cache, which a user's phone or computer does not have.
//!
//! An action normally exports `process`, `shapecheck`, `get_input_type` and
//! `get_output_type` by those names. Built with `--cfg morflow_pack`, actions
//! keep the functions but drop the exported names, so several can be linked
//! together. The pack library then exports a single [`PackManifest`] through
//! [`morflow_pack!`].

use std::ffi::c_char;

use crate::shapecheck::ShapeCheckFn;
use crate::{GetTypeFn, ProcessFn};

/// Layout version of [`PackManifest`]; the engine rejects other versions.
pub const PACK_ABI_VERSION: u32 = 1;

/// The symbol a pack library exports, of type [`PackManifestFn`].
pub const PACK_MANIFEST_SYMBOL: &[u8] = b"morflow_pack_manifest";

/// Returns the pack's manifest, which lives as long as the library.
pub type PackManifestFn = extern "C" fn() -> *const PackManifest;

/// One action of a pack: its name and entry points.
#[repr(C)]
pub struct PackAction {
    /// NUL-terminated action name, such as `identity`.
    pub name: *const c_char,
    pub get_input_type: GetTypeFn,
    pub get_output_type: GetTypeFn,
    pub process: ProcessFn,
    pub shapecheck: ShapeCheckFn,
}

/// Everything a pack library contains.
#[repr(C)]
pub struct PackManifest {
    pub abi_version: u32,
    /// NUL-terminated pack name, such as `base`.
    pub pack: *const c_char,
    /// NUL-terminated pack version, such as `0.3.3`.
    pub version: *const c_char,
    pub actions: *const PackAction,
    pub action_count: usize,
}

// Manifests are immutable statics holding pointers to string literals and
// functions, so sharing them between threads is safe.
unsafe impl Sync for PackAction {}
unsafe impl Sync for PackManifest {}

/// The library name of `pack`'s pack library, without platform prefix or
/// extension (`libmorflow_pack_base.so`, `morflow_pack_base.framework`).
pub fn pack_library_name(pack: &str) -> String {
    format!("morflow_pack_{pack}")
}

/// Declares a pack library's manifest. Each action crate must be built with
/// `--cfg morflow_pack`.
///
/// ```ignore
/// morflow_pack! {
///     pack: "base",
///     version: "0.3.3",
///     actions: ["identity" => identity, "to_tensor" => to_tensor],
/// }
/// ```
#[macro_export]
macro_rules! morflow_pack {
    (pack: $pack:literal, version: $version:literal, actions: [$($name:literal => $krate:ident),* $(,)?] $(,)?) => {
        static MORFLOW_PACK_ACTIONS: &[$crate::pack::PackAction] = &[$(
            $crate::pack::PackAction {
                name: concat!($name, "\0").as_ptr().cast(),
                get_input_type: $krate::get_input_type,
                get_output_type: $krate::get_output_type,
                process: $krate::process,
                shapecheck: $krate::shapecheck,
            }
        ),*];

        static MORFLOW_PACK_MANIFEST: $crate::pack::PackManifest = $crate::pack::PackManifest {
            abi_version: $crate::pack::PACK_ABI_VERSION,
            pack: concat!($pack, "\0").as_ptr().cast(),
            version: concat!($version, "\0").as_ptr().cast(),
            actions: MORFLOW_PACK_ACTIONS.as_ptr(),
            action_count: MORFLOW_PACK_ACTIONS.len(),
        };

        #[no_mangle]
        pub extern "C" fn morflow_pack_manifest() -> *const $crate::pack::PackManifest {
            &MORFLOW_PACK_MANIFEST
        }
    };
}
