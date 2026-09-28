use std::collections::HashMap;
use std::env;
use std::env::consts::{DLL_EXTENSION, DLL_PREFIX};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use core_types::{DataType, GetTypeFn, Payload, ProcessFn};
use libloading::{Library, Symbol};

/// A compiled, dynamically loaded action plugin kept warm in memory.
pub struct LoadedPlugin {
    pub name: String,
    pub path: PathBuf,
    pub input_type: DataType,
    pub output_type: DataType,
    process_fn: ProcessFn,
    // Keeps library handle alive in memory so function pointers remain valid
    _library: Arc<Library>,
}

unsafe impl Send for LoadedPlugin {}
unsafe impl Sync for LoadedPlugin {}

impl LoadedPlugin {
    /// Dispatches a payload to the dynamic library's `process` function via FFI.
    #[inline]
    pub fn process(&self, payload: Payload) -> Payload {
        (self.process_fn)(payload)
    }
}

/// Thread-safe in-memory cache and resolver for Morflow action plugins.
/// Eliminates dynamic library open (`dlopen`) and symbol lookup (`dlsym`) overhead during pipeline execution.
pub struct PluginRegistry {
    search_paths: Vec<PathBuf>,
    cache: RwLock<HashMap<String, Arc<LoadedPlugin>>>,
}

unsafe impl Send for PluginRegistry {}
unsafe impl Sync for PluginRegistry {}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new(Self::default_search_paths())
    }
}

impl PluginRegistry {
    /// Creates a new plugin registry with specified search directories.
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

    /// Normalizes pack name aliases (e.g. image_essential <-> image_essentials)
    pub fn pack_aliases(pack: &str) -> Vec<String> {
        let mut aliases = vec![pack.to_string()];
        if pack == "image_essential" {
            aliases.push("image_essentials".to_string());
        } else if pack == "image_essentials" {
            aliases.push("image_essential".to_string());
        } else if pack == "audio_essential" {
            aliases.push("audio_essentials".to_string());
        } else if pack == "audio_essentials" {
            aliases.push("audio_essential".to_string());
        }
        aliases
    }

    /// Locates the shared library file for the given action in a specific ActionPack.
    pub fn find_action_in_pack(&self, pack: &str, action_name: &str) -> Option<PathBuf> {
        let pack_candidates = Self::pack_aliases(pack);
        let file_candidates = [
            format!("{}_action.{}", action_name, DLL_EXTENSION),
            format!("{}{}_action.{}", DLL_PREFIX, action_name, DLL_EXTENSION),
            format!("{}{}.{}", DLL_PREFIX, action_name, DLL_EXTENSION),
            format!("{}.{}", action_name, DLL_EXTENSION),
        ];

        for base_dir in &self.search_paths {
            for pack_cand in &pack_candidates {
                let pack_dir = base_dir.join(pack_cand);
                for candidate in &file_candidates {
                    let full = pack_dir.join(candidate);
                    if full.is_file() {
                        return Some(full);
                    }
                }
            }
        }

        // Also check direct search paths as fallback
        for base_dir in &self.search_paths {
            for candidate in &file_candidates {
                let full = base_dir.join(candidate);
                if full.is_file() {
                    return Some(full);
                }
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

        let file_candidates = [
            format!("{}_action.{}", action_name, DLL_EXTENSION),
            format!("{}{}_action.{}", DLL_PREFIX, action_name, DLL_EXTENSION),
            format!("{}{}.{}", DLL_PREFIX, action_name, DLL_EXTENSION),
            format!("{}.{}", action_name, DLL_EXTENSION),
        ];

        // 1. Search in known ActionPack subdirectories
        let known_packs = [
            "base",
            "image_essentials",
            "audio_essentials",
            "image_essential",
            "audio_essential",
        ];
        for base_dir in &self.search_paths {
            for pack in &known_packs {
                let pack_dir = base_dir.join(pack);
                for candidate in &file_candidates {
                    let full = pack_dir.join(candidate);
                    if full.is_file() {
                        return Some(full);
                    }
                }
            }
        }

        // 2. Search directly in search_paths
        for base_dir in &self.search_paths {
            for candidate in &file_candidates {
                let full = base_dir.join(candidate);
                if full.is_file() {
                    return Some(full);
                }
            }
        }

        // 3. Search all subdirectories of search_paths (for custom ActionPacks)
        for base_dir in &self.search_paths {
            if let Ok(entries) = std::fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let sub_dir = entry.path();
                        for candidate in &file_candidates {
                            let full = sub_dir.join(candidate);
                            if full.is_file() {
                                return Some(full);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Loads or returns a cached plugin from a specific ActionPack.
    pub fn get_or_load_in_pack(
        &self,
        pack: &str,
        action_name: &str,
    ) -> Result<Arc<LoadedPlugin>, String> {
        let key = format!("{}.{}", pack, action_name);
        {
            let guard = self.cache.read().unwrap();
            if let Some(plugin) = guard.get(&key) {
                return Ok(Arc::clone(plugin));
            }
            if let Some(plugin) = guard.get(action_name) {
                return Ok(Arc::clone(plugin));
            }
        }

        let path = self.find_action_in_pack(pack, action_name).ok_or_else(|| {
            format!(
                "Action '{}' in pack '{}' not found in search paths: {:?}",
                action_name, pack, self.search_paths
            )
        })?;

        let loaded = self.load_from_path(action_name, &path)?;
        let arc_plugin = Arc::new(loaded);

        let mut write_guard = self.cache.write().unwrap();
        write_guard.insert(key, Arc::clone(&arc_plugin));
        write_guard.insert(action_name.to_string(), Arc::clone(&arc_plugin));

        Ok(arc_plugin)
    }

    /// Returns a cached plugin or loads it from disk, caching the symbols for future calls.
    pub fn get_or_load(&self, action_name: &str) -> Result<Arc<LoadedPlugin>, String> {
        {
            let guard = self.cache.read().unwrap();
            if let Some(plugin) = guard.get(action_name) {
                return Ok(Arc::clone(plugin));
            }
        }

        let path = self.find_action_path(action_name).ok_or_else(|| {
            format!(
                "Action '{}' not found in search paths: {:?}",
                action_name, self.search_paths
            )
        })?;

        let loaded = self.load_from_path(action_name, &path)?;
        let arc_plugin = Arc::new(loaded);

        let mut write_guard = self.cache.write().unwrap();
        write_guard.insert(action_name.to_string(), Arc::clone(&arc_plugin));

        Ok(arc_plugin)
    }

    /// Alias for get_or_load for thread-safe cloned access.
    pub fn get_or_load_cloned(&self, action_name: &str) -> Result<Arc<LoadedPlugin>, String> {
        self.get_or_load(action_name)
    }

    /// Explicitly preloads all action plugins found in the search directories.
    pub fn preload_all(&self) -> Result<usize, String> {
        let mut count = 0;
        let mut found_actions = Vec::new();

        let mut scan_dir = |dir: &Path| {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            let action_name = stem
                                .strip_suffix("_action")
                                .unwrap_or(stem)
                                .strip_prefix(DLL_PREFIX)
                                .unwrap_or(stem);

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
                                        let action_name = stem
                                            .strip_suffix("_action")
                                            .unwrap_or(stem)
                                            .strip_prefix(DLL_PREFIX)
                                            .unwrap_or(stem);

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
    fn load_from_path(&self, action_name: &str, path: &Path) -> Result<LoadedPlugin, String> {
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

            Ok(LoadedPlugin {
                name: action_name.to_string(),
                path: path.to_path_buf(),
                input_type,
                output_type,
                process_fn,
                _library: Arc::new(lib),
            })
        }
    }
}
