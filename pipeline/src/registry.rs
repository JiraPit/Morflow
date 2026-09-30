use crate::artifact::{
    read_verified, snapshot, ActionIdentity, ArtifactReceipt, CacheGuard, ReleaseCatalog,
};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use core_types::{
    ActionArgs, DataType, GetShapeFn, GetTypeFn, Payload, ProcessFn, Shape, ShapeResult,
};
use libloading::{Library, Symbol};

/// A compiled, dynamically loaded action kept warm in memory.
pub struct LoadedAction {
    pub name: String,
    pub identity: ActionIdentity,
    pub receipt: ArtifactReceipt,
    pub path: PathBuf,
    pub input_type: DataType,
    pub output_type: DataType,
    process_fn: ProcessFn,
    /// `get_output_shape` is optional so that actions which cannot report a
    /// shape still load; those simply have an unknown output shape.
    get_shape_fn: Option<GetShapeFn>,
    // Keeps library handle alive in memory so function pointers remain valid
    _library: Arc<Library>,
}

unsafe impl Send for LoadedAction {}
unsafe impl Sync for LoadedAction {}

impl LoadedAction {
    /// Dispatches a payload to the dynamic library's `process` function via FFI.
    #[inline]
    pub fn process(&self, payload: Payload) -> Payload {
        (self.process_fn)(payload)
    }

    /// Asks the action what it would produce for this call.
    ///
    /// An action that does not export a shape fn, or that cannot describe its
    /// result, yields [`ShapeResult::Unknown`] rather than an error, so
    /// callers can always fall back to a wildcard shape.
    pub fn output_result(&self, input: &Shape, args: &ActionArgs) -> ShapeResult {
        match &self.get_shape_fn {
            Some(get_shape) => get_shape(input.clone(), args.clone()),
            None => ShapeResult::Unknown,
        }
    }
}

/// Registry keys include pack, requested version, action, and platform.
/// A registry retains loaded handles; create a new pipeline to refresh latest.
pub struct ActionRegistry {
    search_paths: Vec<PathBuf>,
    cache: RwLock<HashMap<ActionIdentity, Arc<LoadedAction>>>,
}
impl Default for ActionRegistry {
    fn default() -> Self {
        Self::new(Self::default_search_paths())
    }
}
impl ActionRegistry {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            cache: RwLock::new(HashMap::new()),
        }
    }
    pub fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Ok(env_path) = env::var("MORFLOW_ACTIONS_PATH") {
            if !env_path.trim().is_empty() {
                paths.push(PathBuf::from(env_path));
            }
        }

        if let Some(home) = dirs::home_dir() {
            paths.push(home.join(".morflow").join("actions"));
        }

        if let Ok(exe_path) = env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                paths.push(exe_dir.join("actions"));
                paths.push(exe_dir.to_path_buf());
            }
        }

        // Relative paths for dev and release builds (current directory and parent directories)
        let prefixes = ["", "../", "../../", "../../../", "../../../../"];
        for prefix in prefixes {
            paths.push(PathBuf::from(format!("{}target/release/actions", prefix)));
            paths.push(PathBuf::from(format!("{}target/release", prefix)));
            paths.push(PathBuf::from(format!("{}target/debug/actions", prefix)));
            paths.push(PathBuf::from(format!("{}target/debug", prefix)));
        }

        paths
    }

    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().into());
    }
    pub fn catalog(&self, pack: &str, version: &str) -> Result<Vec<String>, String> {
        for root in &self.search_paths {
            if ReleaseCatalog::path(root, pack, version).is_file() {
                return Ok(ReleaseCatalog::read(root, pack, version)?
                    .actions(crate::cli::get_host_platform().0));
            }
        }
        Err(format!(
            "No prepared catalog for {pack}/{version}; run morflow prep or install"
        ))
    }
    pub fn get_or_load(&self, identity: &ActionIdentity) -> Result<Arc<LoadedAction>, String> {
        let canonical = ActionIdentity::for_platform(
            &identity.pack,
            &identity.version,
            &identity.action,
            &identity.platform,
        )?;
        if canonical != *identity {
            return Err("Noncanonical action identity".into());
        }
        if identity.platform != crate::cli::get_host_platform().0 {
            return Err(format!(
                "Cannot load {identity} on {}",
                crate::cli::get_host_platform().0
            ));
        }
        // Serialize first loads so handles cannot race against refreshes.
        let mut cache = self.cache.write().map_err(|e| e.to_string())?;
        if let Some(action) = cache.get(identity) {
            return Ok(Arc::clone(action));
        }
        for root in &self.search_paths {
            if identity.path(root).exists()
                || crate::artifact::receipt_path(&identity.path(root)).exists()
            {
                let _maintenance = CacheGuard::shared(root, "maintenance")?;
                let _guard = CacheGuard::acquire(root, &identity.to_string())?;
                let (receipt, bytes) = read_verified(root, identity)?;
                let path = snapshot(root, identity, &bytes, &receipt.sha256)?;
                let action = Arc::new(self.load_from_path(identity, receipt, &path)?);
                cache.insert(identity.clone(), Arc::clone(&action));
                return Ok(action);
            }
        }
        Err(format!(
            "Verified binary for {identity} is missing; run morflow prep or install"
        ))
    }
    /// Low-level loader that inspects and caches symbols from a `.so` / `.dll` file.
    fn load_from_path(
        &self,
        identity: &ActionIdentity,
        receipt: ArtifactReceipt,
        path: &Path,
    ) -> Result<LoadedAction, String> {
        let action_name = &identity.action;
        unsafe {
            let lib = Library::new(path).map_err(|e| {
                format!(
                    "Failed to dlopen '{}' ({}): {}",
                    action_name,
                    path.display(),
                    e
                )
            })?;

            let get_in_sym: Symbol<GetTypeFn> = lib.get(b"get_input_type").map_err(|e| {
                format!(
                    "Missing 'get_input_type' symbol in '{}': {}",
                    path.display(),
                    e
                )
            })?;
            let get_out_sym: Symbol<GetTypeFn> = lib.get(b"get_output_type").map_err(|e| {
                format!(
                    "Missing 'get_output_type' symbol in '{}': {}",
                    path.display(),
                    e
                )
            })?;
            let process_sym: Symbol<ProcessFn> = lib
                .get(b"process")
                .map_err(|e| format!("Missing 'process' symbol in '{}': {}", path.display(), e))?;

            let input_type = (*get_in_sym)();
            let output_type = (*get_out_sym)();
            let process_fn = *process_sym;
            // The exported shape fn. A missing symbol simply means the action cannot
            // describe its output and reports an unknown shape.
            let get_shape_fn: Option<GetShapeFn> = lib.get(b"get_output_shape").ok().map(|s| *s);
            Ok(LoadedAction {
                name: action_name.to_string(),
                identity: identity.clone(),
                receipt,
                path: path.to_path_buf(),
                input_type,
                output_type,
                process_fn,
                get_shape_fn,
                _library: Arc::new(lib),
            })
        }
    }
}
