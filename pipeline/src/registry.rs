use std::collections::HashMap;
use std::env;
use std::env::consts::DLL_EXTENSION;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use core_types::{
    ActionArgs, DataType, GetShapeFn, GetTypeFn, Payload, ProcessFn, Shape, ShapeResult,
};
use libloading::{Library, Symbol};

/// A compiled, dynamically loaded action kept warm in memory.
pub struct LoadedAction {
    pub name: String,
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

/// Thread-safe in-memory cache and resolver for Morflow actions.
/// Eliminates dynamic library open (`dlopen`) and symbol lookup (`dlsym`) overhead during pipeline execution.
pub struct ActionRegistry {
    search_paths: Vec<PathBuf>,
    cache: RwLock<HashMap<String, Arc<LoadedAction>>>,
}

unsafe impl Send for ActionRegistry {}
unsafe impl Sync for ActionRegistry {}

impl Default for ActionRegistry {
    fn default() -> Self {
        Self::new(Self::default_search_paths())
    }
}

impl ActionRegistry {
    /// Creates a new action registry with specified search directories.
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Standard search paths including environment variables, binary directory, and target folders.
    pub fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Ok(env_path) = env::var("MORFLOW_ACTIONS_PATH") {
            paths.push(PathBuf::from(env_path));
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

    /// Adds an additional search directory to the registry.
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Locates the shared library file for the given action in a specific ActionPack.
    pub fn find_action_in_pack(&self, pack: &str, action_name: &str) -> Option<PathBuf> {
        let file_candidate = format!("{}_action.{}", action_name, DLL_EXTENSION);

        for base_dir in &self.search_paths {
            let full = base_dir.join(pack).join(&file_candidate);
            if full.is_file() {
                return Some(full);
            }
        }

        // Also check direct search paths
        for base_dir in &self.search_paths {
            let full = base_dir.join(&file_candidate);
            if full.is_file() {
                return Some(full);
            }
        }

        None
    }

    /// Locates the shared library file for the given action name on disk.
    pub fn find_action_path(&self, action_name: &str) -> Option<PathBuf> {
        if let Some((pack, act)) = action_name.split_once('.') {
            if let Some(path) = self.find_action_in_pack(pack, act) {
                return Some(path);
            }
        }

        let file_candidate = format!("{}_action.{}", action_name, DLL_EXTENSION);

        // 1. Search in known ActionPack subdirectories
        let known_packs = [
            "base",
            "image_essentials",
            "audio_essentials",
            "tensor_essentials",
            "math_essentials",
            "tensor_stats",
            "nn_essentials",
            "linalg_essentials",
        ];
        for base_dir in &self.search_paths {
            for pack in &known_packs {
                let full = base_dir.join(pack).join(&file_candidate);
                if full.is_file() {
                    return Some(full);
                }
            }
        }

        // 2. Search directly in search_paths
        for base_dir in &self.search_paths {
            let full = base_dir.join(&file_candidate);
            if full.is_file() {
                return Some(full);
            }
        }

        // 3. Search all subdirectories of search_paths (for custom ActionPacks)
        for base_dir in &self.search_paths {
            if let Ok(entries) = std::fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let sub_dir = entry.path();
                        let full = sub_dir.join(&file_candidate);
                        if full.is_file() {
                            return Some(full);
                        }
                    }
                }
            }
        }

        None
    }

    /// Loads or returns a cached action from a specific ActionPack.
    pub fn get_or_load_in_pack(
        &self,
        pack: &str,
        action_name: &str,
    ) -> Result<Arc<LoadedAction>, String> {
        let key = format!("{}.{}", pack, action_name);
        {
            let guard = self.cache.read().unwrap();
            if let Some(action) = guard.get(&key) {
                return Ok(Arc::clone(action));
            }
            if let Some(action) = guard.get(action_name) {
                return Ok(Arc::clone(action));
            }
        }

        let path = self.find_action_in_pack(pack, action_name).ok_or_else(|| {
            format!(
                "Action '{}' in pack '{}' not found in search paths: {:?}",
                action_name, pack, self.search_paths
            )
        })?;

        let loaded = self.load_from_path(action_name, &path)?;
        let arc_action = Arc::new(loaded);

        let mut write_guard = self.cache.write().unwrap();
        write_guard.insert(key, Arc::clone(&arc_action));
        write_guard.insert(action_name.to_string(), Arc::clone(&arc_action));

        Ok(arc_action)
    }

    /// Returns a cached action or loads it from disk, caching the symbols for future calls.
    pub fn get_or_load(&self, action_name: &str) -> Result<Arc<LoadedAction>, String> {
        {
            let guard = self.cache.read().unwrap();
            if let Some(action) = guard.get(action_name) {
                return Ok(Arc::clone(action));
            }
        }

        let path = self.find_action_path(action_name).ok_or_else(|| {
            format!(
                "Action '{}' not found in search paths: {:?}",
                action_name, self.search_paths
            )
        })?;

        let loaded = self.load_from_path(action_name, &path)?;
        let arc_action = Arc::new(loaded);

        let mut write_guard = self.cache.write().unwrap();
        write_guard.insert(action_name.to_string(), Arc::clone(&arc_action));

        Ok(arc_action)
    }

    /// Alias for get_or_load for thread-safe cloned access.
    pub fn get_or_load_cloned(&self, action_name: &str) -> Result<Arc<LoadedAction>, String> {
        self.get_or_load(action_name)
    }

    /// Explicitly preloads all actions found in the search directories.
    pub fn preload_all(&self) -> Result<usize, String> {
        let mut count = 0;
        let mut found_actions = Vec::new();

        let mut scan_dir = |dir: &Path| {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            let action_name = stem.strip_suffix("_action").unwrap_or(stem);

                            if !found_actions.contains(&action_name.to_string()) {
                                found_actions.push(action_name.to_string());
                            }
                        }
                    } else if path.is_dir() {
                        if let Ok(sub_entries) = std::fs::read_dir(&path) {
                            for sub_entry in sub_entries.flatten() {
                                let sub_path = sub_entry.path();
                                if sub_path.is_file() {
                                    if let Some(stem) =
                                        sub_path.file_stem().and_then(|s| s.to_str())
                                    {
                                        let action_name =
                                            stem.strip_suffix("_action").unwrap_or(stem);

                                        if !found_actions.contains(&action_name.to_string()) {
                                            found_actions.push(action_name.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        for dir in &self.search_paths {
            scan_dir(dir);
        }

        for action_name in found_actions {
            if self.get_or_load(&action_name).is_ok() {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Low-level loader that inspects and caches symbols from a `.so` / `.dll` file.
    fn load_from_path(&self, action_name: &str, path: &Path) -> Result<LoadedAction, String> {
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
