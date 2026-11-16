//! Actions bundled into an app as pack libraries (see `core_types::pack`).
//!
//! Android and iOS apps cannot load actions downloaded into the action cache,
//! so they ship one pack library per pack inside the app instead. The engine
//! falls back to these when an action is not in the cache. The operating
//! system verifies bundled code when the app is installed, so there are no
//! receipts or checksums to check.
//!
//! Pack libraries are found:
//! - in the directories listed in `MORFLOW_PACKS_PATH` (desktop and tests);
//! - on Android, by library name, among the app's native libraries;
//! - on iOS, as frameworks next to the engine's own framework.

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use core_types::pack::{pack_library_name, PackManifestFn, PACK_ABI_VERSION, PACK_MANIFEST_SYMBOL};
use core_types::{DataType, GetTypeFn, ProcessFn, ShapeCheckFn};
use libloading::Library;

/// One action of a loaded pack library.
#[derive(Clone, Copy)]
pub(crate) struct BundledAction {
    pub input_type: DataType,
    pub output_type: DataType,
    pub process: ProcessFn,
    pub shapecheck: ShapeCheckFn,
}

/// A loaded pack library.
pub(crate) struct BundledPack {
    pub version: String,
    pub path: PathBuf,
    pub library: Arc<Library>,
    actions: HashMap<String, BundledAction>,
}

impl BundledPack {
    /// Action names in the pack, sorted.
    pub fn action_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.actions.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn action(&self, name: &str) -> Option<BundledAction> {
        self.actions.get(name).copied()
    }

    /// Whether `requested` (an exact version or `latest`) selects this pack.
    pub fn serves(&self, requested: &str) -> bool {
        requested == "latest" || requested == self.version
    }
}

fn library_file(stem: &str) -> String {
    let (_, extension) = crate::platform::host_platform();
    if cfg!(windows) {
        format!("{stem}.{extension}")
    } else {
        format!("lib{stem}.{extension}")
    }
}

/// Where `pack`'s library may be, most specific first.
fn candidates(pack: &str) -> Vec<PathBuf> {
    let stem = pack_library_name(pack);
    let mut paths = Vec::new();
    if let Some(dirs) = std::env::var_os("MORFLOW_PACKS_PATH") {
        for dir in std::env::split_paths(&dirs).filter(|d| !d.as_os_str().is_empty()) {
            paths.push(dir.join(library_file(&stem)));
        }
    }
    if cfg!(target_os = "android") {
        // A bare name makes the dynamic linker search the app's libraries.
        paths.push(PathBuf::from(library_file(&stem)));
    }
    if cfg!(target_os = "ios") {
        // Flutter bundles libmorflow_pack_<pack>.dylib as
        // Frameworks/morflow_pack_<pack>.framework/morflow_pack_<pack>.
        if let Some(frameworks) = own_frameworks_dir() {
            paths.push(frameworks.join(format!("{stem}.framework")).join(&stem));
        }
    }
    paths
}

/// The directory holding the framework this engine was loaded from, found
/// from the address of one of its own functions.
#[cfg(unix)]
fn own_frameworks_dir() -> Option<PathBuf> {
    let mut info: libc_dl::DlInfo = unsafe { std::mem::zeroed() };
    let found = unsafe { libc_dl::dladdr(own_frameworks_dir as *const _, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return None;
    }
    let binary = PathBuf::from(unsafe { CStr::from_ptr(info.dli_fname) }.to_str().ok()?);
    // .../Frameworks/<engine>.framework/<engine> -> .../Frameworks
    binary.parent()?.parent().map(Path::to_path_buf)
}

#[cfg(not(unix))]
fn own_frameworks_dir() -> Option<PathBuf> {
    None
}

#[cfg(unix)]
mod libc_dl {
    use std::ffi::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct DlInfo {
        pub dli_fname: *const c_char,
        pub dli_fbase: *mut c_void,
        pub dli_sname: *const c_char,
        pub dli_saddr: *mut c_void,
    }

    extern "C" {
        pub fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }
}

unsafe fn read(path: &Path, library: Arc<Library>, pack: &str) -> Result<BundledPack, String> {
    let manifest_fn = *library
        .get::<PackManifestFn>(PACK_MANIFEST_SYMBOL)
        .map_err(|e| format!("{} is not a pack library: {e}", path.display()))?;
    let manifest = &*manifest_fn();
    if manifest.abi_version != PACK_ABI_VERSION {
        return Err(format!(
            "{} has pack format {}, but this engine reads format {}",
            path.display(),
            manifest.abi_version,
            PACK_ABI_VERSION
        ));
    }
    let name = CStr::from_ptr(manifest.pack).to_string_lossy();
    if name != pack {
        return Err(format!("{} contains pack {name}, not {pack}", path.display()));
    }
    let actions = std::slice::from_raw_parts(manifest.actions, manifest.action_count)
        .iter()
        .map(|action| {
            let get_input_type: GetTypeFn = action.get_input_type;
            let get_output_type: GetTypeFn = action.get_output_type;
            (
                CStr::from_ptr(action.name).to_string_lossy().into_owned(),
                BundledAction {
                    input_type: get_input_type(),
                    output_type: get_output_type(),
                    process: action.process,
                    shapecheck: action.shapecheck,
                },
            )
        })
        .collect();
    Ok(BundledPack {
        version: CStr::from_ptr(manifest.version).to_string_lossy().into_owned(),
        path: path.to_path_buf(),
        library,
        actions,
    })
}

/// The bundled pack library for `pack`: `Ok(None)` when it is not bundled,
/// or an error when a library was found but cannot be used.
pub(crate) fn find(pack: &str) -> Result<Option<Arc<BundledPack>>, String> {
    static PACKS: OnceLock<Mutex<HashMap<String, Arc<BundledPack>>>> = OnceLock::new();
    let mut packs = PACKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| e.to_string())?;
    if let Some(found) = packs.get(pack) {
        return Ok(Some(Arc::clone(found)));
    }
    for path in candidates(pack) {
        // A bare name is resolved by the dynamic linker; a path must exist.
        if path.components().count() > 1 && !path.exists() {
            continue;
        }
        let Ok(library) = crate::native::library(&path) else {
            continue;
        };
        let loaded = Arc::new(unsafe { read(&path, library, pack) }?);
        packs.insert(pack.to_string(), Arc::clone(&loaded));
        return Ok(Some(loaded));
    }
    Ok(None)
}
