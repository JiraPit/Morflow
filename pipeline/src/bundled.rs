//! Actions bundled into an app as pack libraries (see `core_types::pack`).
//!
//! An app ships one pack library per pack it uses, so its pipelines run
//! without the action cache: Android and iOS apps cannot download actions,
//! and a desktop app's users have no Morflow CLI. A bundled pack serves the
//! actions of its version (and `latest`) before the action cache does. The
//! operating system or the app's installer verifies bundled code, so there are
//! no receipts or checksums to check.
//!
//! Pack libraries are found:
//! - in the directories listed in `MORFLOW_PACKS_PATH`;
//! - on Android, by library name, among the app's native libraries;
//! - next to the library the engine itself was loaded from, where Dart and
//!   Flutter put every bundled native library on Linux and Windows;
//! - on macOS and iOS, as frameworks in the `Frameworks` directory that holds
//!   the engine's own framework.

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
    } else if let Some(engine) = own_library() {
        paths.extend(beside_engine(&engine, &stem, &library_file(&stem)));
    }
    paths
}

/// Where an app bundles `stem` relative to the engine library at `engine`.
fn beside_engine(engine: &Path, stem: &str, file: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(dir) = engine.parent() {
        paths.push(dir.join(file));
    }
    if cfg!(target_vendor = "apple") {
        // Flutter bundles libX.dylib as Frameworks/X.framework/X (on macOS
        // through Versions/A), so look in the nearest Frameworks directory.
        if let Some(frameworks) = engine
            .ancestors()
            .find(|dir| dir.file_name().is_some_and(|n| n == "Frameworks"))
        {
            paths.push(frameworks.join(format!("{stem}.framework")).join(stem));
        }
    }
    paths
}

/// The file this engine was loaded from, found from the address of one of
/// its own functions.
#[cfg(unix)]
fn own_library() -> Option<PathBuf> {
    let mut info: libc_dl::DlInfo = unsafe { std::mem::zeroed() };
    let found = unsafe { libc_dl::dladdr(own_library as *const _, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return None;
    }
    let path = PathBuf::from(unsafe { CStr::from_ptr(info.dli_fname) }.to_str().ok()?);
    // A program linking the engine statically reports its own path, which may
    // be relative to where it started; bundled libraries are found by path.
    path.is_absolute().then_some(path)
}

#[cfg(windows)]
fn own_library() -> Option<PathBuf> {
    use std::ffi::{c_void, OsString};
    use std::os::windows::ffi::OsStringExt;

    const FROM_ADDRESS: u32 = 0x4;
    const UNCHANGED_REFCOUNT: u32 = 0x2;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleExW(flags: u32, address: *const c_void, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
    }
    let mut module = std::ptr::null_mut();
    let found = unsafe {
        GetModuleHandleExW(
            FROM_ADDRESS | UNCHANGED_REFCOUNT,
            own_library as *const c_void,
            &mut module,
        )
    };
    if found == 0 {
        return None;
    }
    let mut name = vec![0u16; 32_768];
    let length =
        unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
    if length == 0 || length >= name.len() {
        return None;
    }
    Some(PathBuf::from(OsString::from_wide(&name[..length])))
}

#[cfg(not(any(unix, windows)))]
fn own_library() -> Option<PathBuf> {
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
        return Err(format!(
            "{} contains pack {name}, not {pack}",
            path.display()
        ));
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
        version: CStr::from_ptr(manifest.version)
            .to_string_lossy()
            .into_owned(),
        path: path.to_path_buf(),
        library,
        actions,
    })
}

/// The bundled pack library for `pack`: `Ok(None)` when it is not bundled,
/// or an error when a library was found but cannot be used.
pub(crate) fn find(pack: &str) -> Result<Option<Arc<BundledPack>>, String> {
    // Looked up once per pack: an app's bundled libraries do not change while
    // it runs, and most packs are not bundled outside apps.
    static PACKS: OnceLock<Mutex<HashMap<String, Option<Arc<BundledPack>>>>> = OnceLock::new();
    let mut packs = PACKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| e.to_string())?;
    if let Some(found) = packs.get(pack) {
        return Ok(found.clone());
    }
    let mut found = None;
    for path in candidates(pack) {
        // A bare name is resolved by the dynamic linker; a path must exist.
        if path.components().count() > 1 && !path.exists() {
            continue;
        }
        let Ok(library) = crate::native::library(&path) else {
            continue;
        };
        found = Some(Arc::new(unsafe { read(&path, library, pack) }?));
        break;
    }
    packs.insert(pack.to_string(), found.clone());
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_are_looked_up_beside_the_engine() {
        let file = library_file("morflow_pack_base");
        let engine = Path::new("/app/lib").join(library_file("morflow_capi"));
        let paths = beside_engine(&engine, "morflow_pack_base", &file);
        assert_eq!(paths[0], Path::new("/app/lib").join(&file));
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn packs_are_looked_up_among_frameworks() {
        for engine in [
            "/A.app/Frameworks/morflow_capi.framework/morflow_capi",
            "/A.app/Contents/Frameworks/morflow_capi.framework/Versions/A/morflow_capi",
        ] {
            let paths = beside_engine(
                Path::new(engine),
                "morflow_pack_base",
                "libmorflow_pack_base.dylib",
            );
            let frameworks = engine.split("/morflow_capi.framework").next().unwrap();
            assert_eq!(
                paths.last().unwrap(),
                &Path::new(frameworks).join("morflow_pack_base.framework/morflow_pack_base")
            );
        }
    }

    #[test]
    fn the_engine_knows_its_own_library() {
        let path = own_library().expect("the test binary's path");
        assert!(path.is_absolute() && path.exists(), "{}", path.display());
    }
}
