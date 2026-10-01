use crate::artifact::{
    read_verified, snapshot, ActionIdentity, ArtifactReceipt, CacheGuard, ReleaseCatalog,
};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use core_types::{
    ActionArgs, DataType, GetTypeFn, OutputComponent, Payload, ProcessFn, RVec, Shape, ShapeResult,
    ValueShape, ValueShapeResult,
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
    /// Required shape analysis and invocation-local preparation callback.
    shapecheck_fn: core_types::ShapeCheckFn,
    // Keeps library handle alive in memory so function pointers remain valid
    _library: Arc<Library>,
}

unsafe impl Send for LoadedAction {}
unsafe impl Sync for LoadedAction {}

impl LoadedAction {
    pub fn shapecheck(
        &self,
        input: core_types::InputDescriptor,
        args: ActionArgs,
    ) -> core_types::ShapeCheckResult {
        (self.shapecheck_fn)(input, args)
    }
    /// The only runtime dispatch path: mandatory preparation and verification.
    pub fn process(&self, payload: Payload) -> Payload {
        core_types::shapecheck::execute(&self.name, self.shapecheck_fn, self.process_fn, payload)
    }

    pub fn output_value_result(
        &self,
        input: &ValueShape,
        args: &ActionArgs,
    ) -> Option<ValueShapeResult> {
        Some(
            self.shapecheck(
                core_types::InputDescriptor::partial(input.clone()),
                args.clone(),
            )
            .prediction(),
        )
    }
    pub fn output_result(&self, input: &Shape, args: &ActionArgs) -> ShapeResult {
        let kind = if self.input_type.intersects(DataType::Tensor) {
            DataType::Tensor
        } else {
            self.input_type
        };
        match self
            .output_value_result(
                &ValueShape::Leaf {
                    kind,
                    shape: Some(input.clone()).into(),
                },
                args,
            )
            .unwrap()
        {
            ValueShapeResult::Invalid(reason) => ShapeResult::Invalid(reason),
            ValueShapeResult::Ok(output) => output
                .shape()
                .cloned()
                .map_or(ShapeResult::Unknown, ShapeResult::Ok),
            ValueShapeResult::Unknown => ShapeResult::Unknown,
        }
    }
    pub fn output_components(
        &self,
        input: &Shape,
        args: &ActionArgs,
    ) -> Option<RVec<OutputComponent>> {
        let ValueShapeResult::Ok(ValueShape::Composite(items)) =
            self.output_value_result(&ValueShape::tensor(input.clone()), args)?
        else {
            return None;
        };
        Some(
            items
                .iter()
                .map(|item| match item {
                    ValueShape::Leaf { kind, shape } => OutputComponent {
                        kind: *kind,
                        shape: shape
                            .as_ref()
                            .into_option()
                            .cloned()
                            .map_or(ShapeResult::Unknown, ShapeResult::Ok),
                    },
                    _ => OutputComponent {
                        kind: DataType::Composite,
                        shape: ShapeResult::Unknown,
                    },
                })
                .collect(),
        )
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

            verify_action_abi(&lib)?;
            let shapecheck_fn = *lib
                .get::<core_types::ShapeCheckFn>(b"shapecheck")
                .map_err(|e| format!("Missing mandatory shapecheck export: {e}"))?;
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
            Ok(LoadedAction {
                name: action_name.to_string(),
                identity: identity.clone(),
                receipt,
                path: path.to_path_buf(),
                input_type,
                output_type,
                process_fn,
                shapecheck_fn,
                _library: Arc::new(lib),
            })
        }
    }
}

fn verify_action_abi(lib: &Library) -> Result<(), String> {
    unsafe {
        let version = lib
            .get::<extern "C" fn() -> u32>(b"get_action_abi_version")
            .map_err(|_| {
                "Action lacks the mandatory shapecheck ABI. Prepare an updated action version."
                    .to_string()
            })?;
        if version() != core_types::shapecheck::ACTION_ABI_VERSION {
            return Err("Incompatible action ABI. Prepare an updated action version.".into());
        }
        let layout = lib
            .get::<extern "C" fn() -> *const core_types::abi_stable::type_layout::TypeLayout>(
                b"get_action_abi_layout",
            )
            .map_err(|_| "Missing action ABI layout export".to_string())?;
        let actual = layout().as_ref().ok_or("Null action ABI layout")?;
        core_types::abi_stable::abi_stability::check_layout_compatibility(
            <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT,
            actual,
        )
        .map_err(|e| format!("Incompatible action ABI layout: {e}"))
        .into()
    }
}
#[cfg(all(test, unix))]
mod shape_abi_tests {
    use super::*;
    #[test]
    fn layouts_and_mandatory_shapecheck_are_checked_before_type_callbacks() {
        let temp = tempfile::tempdir().unwrap();
        let expected = <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT;
        let wrong = <Shape as core_types::StableAbi>::LAYOUT;
        let identity = ActionIdentity::new("test_pack", "0.3.0", "fixture").unwrap();
        for (case, layout) in [
            ("null", None),
            ("wrong", Some(wrong)),
            ("missing_shapecheck", Some(expected)),
        ] {
            let source = temp.path().join(format!("{case}.c"));
            let path = source.with_extension("so");
            let address = layout.map_or(0, |layout| layout as *const _ as usize);
            std::fs::write(&source, format!(
                "#include <stdlib.h>\nunsigned get_action_abi_version(void) {{return 1;}}\nconst void *get_action_abi_layout(void) {{return (const void *)0x{address:x};}}\nvoid get_input_type(void) {{abort();}}\nvoid get_output_type(void) {{abort();}}\nvoid process(void) {{abort();}}\n"
            )).unwrap();
            assert!(std::process::Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&path)
                .status()
                .unwrap()
                .success());
            let receipt = ArtifactReceipt {
                identity: identity.clone(),
                concrete_version: "0.3.0".into(),
                repository: "local/test".into(),
                sha256: "unused".into(),
            };
            let error = ActionRegistry::default()
                .load_from_path(&identity, receipt, &path)
                .err()
                .expect("fixture must be rejected");
            assert!(
                error.contains(match case {
                    "null" => "Null action ABI layout",
                    "wrong" => "Incompatible action ABI layout",
                    _ => "shapecheck",
                }),
                "{error}"
            );
        }
    }
    #[test]
    fn legacy_and_incompatible_actions_are_rejected_before_callbacks() {
        let temp = tempfile::tempdir().unwrap();
        for version in [None, Some(0), Some(1), Some(99)] {
            let source = temp.path().join(format!("version{version:?}.c"));
            let library = source.with_extension("so");
            let marker = version.map_or(String::new(), |v| {
                format!("unsigned get_action_abi_version(void) {{return {v};}}")
            });
            std::fs::write(&source,format!("#include <stdlib.h>\n{marker}\nvoid process(void) {{abort();}}\nvoid shapecheck(void) {{abort();}}\n")).unwrap();
            assert!(std::process::Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success());
            let library = unsafe { Library::new(library) }.unwrap();
            assert!(verify_action_abi(&library).is_err());
        }
    }
}
