//! Verified plugin artifacts and pipeline-scoped, runtime-only library loading.
use crate::artifact::{
    atomic_write, component, digest, normalize_version, read_json, receipt_path, CacheGuard,
};
use core_types::abi_stable::std_types::RResult;
use core_types::plugins::{PluginRequirement, RuntimeContext};
use core_types::{RString, RVec};
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirement {
    pub name: String,
    pub version: String,
}
impl Requirement {
    pub fn validate(&self) -> Result<(), String> {
        component(&self.name)?;
        semver::VersionReq::parse(&self.version).map_err(|e| {
            format!(
                "Invalid plugin requirement {}/{}: {e}",
                self.name, self.version
            )
        })?;
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginIdentity {
    pub name: String,
    pub version: String,
    pub platform: String,
}
impl PluginIdentity {
    pub fn new(name: &str, version: &str) -> Result<Self, String> {
        component(name)?;
        Ok(Self {
            name: name.into(),
            version: normalize_version(version)?,
            platform: crate::cli::get_host_platform().0.into(),
        })
    }
    pub fn filename(&self) -> String {
        let ext = if self.platform.starts_with("windows-") {
            "dll"
        } else if self.platform.starts_with("darwin-") {
            "dylib"
        } else {
            "so"
        };
        format!(
            "{}_plugin-{}-{}.{}",
            self.name, self.version, self.platform, ext
        )
    }
    pub fn path(&self, root: &Path) -> PathBuf {
        root.join(&self.name).join(self.filename())
    }
}
impl std::fmt::Display for PluginIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{} ({})", self.name, self.version, self.platform)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginReceipt {
    #[serde(flatten)]
    pub identity: PluginIdentity,
    pub concrete_version: String,
    pub repository: String,
    pub sha256: String,
}
impl PluginReceipt {
    pub fn validate(&self, id: &PluginIdentity, bytes: &[u8]) -> Result<(), String> {
        component(&id.name)?;
        component(&id.platform)?;
        if normalize_version(&id.version)? != id.version
            || &self.identity != id
            || self.concrete_version == "latest"
            || normalize_version(&self.concrete_version)? != self.concrete_version
            || (id.version != "latest" && self.concrete_version != id.version)
            || self.repository.is_empty()
            || digest(bytes) != self.sha256
        {
            return Err(format!(
                "Invalid identity or checksum for plugin {id}; run morflow prep"
            ));
        }
        Ok(())
    }
}
pub fn read_verified(root: &Path, id: &PluginIdentity) -> Result<(PluginReceipt, Vec<u8>), String> {
    let path = id.path(root);
    let receipt: PluginReceipt = read_json(&receipt_path(&path))?;
    let bytes =
        std::fs::read(path).map_err(|e| format!("Missing plugin {id}: {e}; run morflow prep"))?;
    receipt.validate(id, &bytes)?;
    Ok((receipt, bytes))
}
pub fn snapshot(
    root: &Path,
    id: &PluginIdentity,
    bytes: &[u8],
    hash: &str,
) -> Result<PathBuf, String> {
    let path = root.join(".objects").join(hash).join(id.filename());
    let _guard = CacheGuard::acquire(root, &format!("plugin-snapshot-{hash}-{}", id.filename()))?;
    if path.exists() {
        if digest(&std::fs::read(&path).map_err(|e| e.to_string())?) != hash {
            return Err(format!("Corrupt plugin snapshot {}", path.display()));
        }
    } else {
        atomic_write(&path, bytes)?;
    }
    std::fs::canonicalize(path).map_err(|e| e.to_string())
}
pub fn cache_dir(custom: Option<PathBuf>) -> PathBuf {
    custom
        .or_else(|| {
            std::env::var_os("MORFLOW_PLUGINS_PATH")
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".morflow/plugins")
        })
}
pub fn search_paths() -> Vec<PathBuf> {
    let mut paths = vec![cache_dir(None)];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            paths.push(parent.join("plugins"));
        }
    }
    for prefix in ["", "../", "../../", "../../../", "../../../../"] {
        for profile in ["release", "debug"] {
            paths.push(PathBuf::from(format!("{prefix}target/{profile}/plugins")));
        }
    }
    paths
}
pub fn declarations(
    decls: &[parser::ast::PluginDecl],
) -> Result<BTreeMap<String, PluginIdentity>, String> {
    let mut result = BTreeMap::new();
    for decl in decls {
        let id = PluginIdentity::new(&decl.name, &decl.version)?;
        if result.insert(id.name.clone(), id).is_some() {
            return Err(format!(
                "Duplicate plugin declaration '{}'; declare one explicit version per plugin",
                decl.name
            ));
        }
    }
    Ok(result)
}
#[derive(Default)]
struct PluginLibrary {
    handle: Option<Arc<Library>>,
    symbols: BTreeMap<String, usize>,
}
struct SelectedPlugin {
    receipt: PluginReceipt,
    path: PathBuf,
    library: Mutex<PluginLibrary>,
}
pub struct PluginSet {
    selected: BTreeMap<String, SelectedPlugin>,
}
impl PluginSet {
    pub fn prepare(
        decls: &[parser::ast::PluginDecl],
        paths: &[PathBuf],
    ) -> Result<Arc<Self>, String> {
        let mut selected = BTreeMap::new();
        for (name, id) in declarations(decls)? {
            let root = paths
                .iter()
                .find(|root| id.path(root).exists() || receipt_path(&id.path(root)).exists())
                .ok_or_else(|| format!("Verified plugin {id} is missing; run morflow prep"))?;
            let _maintenance = CacheGuard::shared(root, "maintenance")?;
            let _guard = CacheGuard::acquire(root, &id.to_string())?;
            let (receipt, bytes) = read_verified(root, &id)?;
            let path = snapshot(root, &id, &bytes, &receipt.sha256)?;
            selected.insert(
                name,
                SelectedPlugin {
                    receipt,
                    path,
                    library: Mutex::new(PluginLibrary::default()),
                },
            );
        }
        Ok(Arc::new(Self { selected }))
    }
    pub fn require(&self, requirements: &[Requirement]) -> Result<(), String> {
        for req in requirements {
            req.validate()?;
            let selected = self.selected.get(&req.name).ok_or_else(|| {
                format!(
                    "Required plugin '{}' is undeclared; add plugin {}/<version> to the pipeline",
                    req.name, req.name
                )
            })?;
            if !semver::VersionReq::parse(&req.version)
                .map_err(|e| e.to_string())?
                .matches(
                    &semver::Version::parse(&selected.receipt.concrete_version)
                        .map_err(|e| e.to_string())?,
                )
            {
                return Err(format!(
                    "Plugin {} requires {}, but pipeline selected {}",
                    req.name, req.version, selected.receipt.concrete_version
                ));
            }
        }
        Ok(())
    }
    pub fn context(self: &Arc<Self>) -> RuntimeContext {
        unsafe {
            RuntimeContext::new(
                Arc::into_raw(self.clone()) as usize,
                release_owner,
                resolve_symbol,
            )
        }
    }
    fn symbol(&self, name: &str, symbol: &str) -> Result<usize, String> {
        if symbol.is_empty() || symbol.contains('\0') {
            return Err("Invalid plugin symbol".into());
        }
        let plugin = self
            .selected
            .get(name)
            .ok_or_else(|| format!("Plugin '{name}' is not declared"))?;
        let mut loaded = plugin.library.lock().map_err(|e| e.to_string())?;
        if let Some(address) = loaded.symbols.get(symbol) {
            return Ok(*address);
        }
        if loaded.handle.is_none() {
            loaded.handle = Some(crate::native::library(&plugin.path).map_err(|e| {
                format!(
                    "Cannot load plugin {}: {e}. Install its required system shared libraries",
                    plugin.receipt.identity
                )
            })?);
        }
        let address = unsafe {
            loaded
                .handle
                .as_ref()
                .unwrap()
                .get::<unsafe extern "C" fn()>(symbol.as_bytes())
                .map(|s| *s as usize)
                .map_err(|e| format!("Plugin {name} has no symbol '{symbol}': {e}"))?
        };
        loaded.symbols.insert(symbol.into(), address);
        Ok(address)
    }
}
extern "C" fn release_owner(owner: usize) {
    // Paired with Arc::into_raw in context().
    unsafe {
        drop(Arc::from_raw(owner as *const PluginSet));
    }
}
extern "C" fn resolve_symbol(
    owner: usize,
    name: RString,
    symbol: RString,
) -> RResult<usize, RString> {
    // The owning Arc is held by LoadedAction throughout every synchronous call.
    let result = std::panic::catch_unwind(|| unsafe {
        (&*(owner as *const PluginSet)).symbol(&name, &symbol)
    });
    result
        .unwrap_or_else(|_| Err("Plugin loading failed".into()))
        .map_err(RString::from)
        .into()
}
pub fn native_requirements(values: RVec<PluginRequirement>) -> Vec<Requirement> {
    values
        .into_iter()
        .map(|r| Requirement {
            name: r.name.to_string(),
            version: r.version.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn publish(root: &Path, version: &str, concrete: &str, bytes: &[u8]) -> PluginIdentity {
        let id = PluginIdentity::new("fixture-bridge", version).unwrap();
        let receipt = PluginReceipt {
            identity: id.clone(),
            concrete_version: concrete.into(),
            repository: "test".into(),
            sha256: digest(bytes),
        };
        let _guard = CacheGuard::acquire(root, &id.to_string()).unwrap();
        atomic_write(&id.path(root), bytes).unwrap();
        atomic_write(
            &receipt_path(&id.path(root)),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        id
    }
    fn decl(version: &str) -> parser::ast::PluginDecl {
        parser::ast::PluginDecl {
            name: "fixture-bridge".into(),
            version: version.into(),
        }
    }
    #[test]
    fn rejects_invalid_versions_duplicate_declarations_and_unprepared_artifacts() {
        assert!(declarations(&[decl("")]).is_err());
        assert!(declarations(&[decl("1")]).is_err());
        assert!(declarations(&[decl("0.1.0"), decl("latest")]).is_err());
        let root = tempfile::tempdir().unwrap();
        assert!(PluginSet::prepare(&[decl("0.1.0")], &[root.path().into()]).is_err());
        let id = publish(root.path(), "0.1.0", "0.1.0", b"verified");
        let set = PluginSet::prepare(&[decl("0.1.0")], &[root.path().into()]).unwrap();
        assert!(set
            .require(&[Requirement {
                name: "fixture-bridge".into(),
                version: "^0.1.0".into()
            }])
            .is_ok());
        assert!(set
            .require(&[Requirement {
                name: "fixture-bridge".into(),
                version: "^0.2.0".into()
            }])
            .is_err());
        assert!(set
            .require(&[Requirement {
                name: "missing".into(),
                version: "^0.1".into()
            }])
            .is_err());
        assert!(set
            .context()
            .symbol("fixture-bridge", "entry")
            .unwrap_err()
            .contains("Cannot load plugin"));
        std::fs::write(id.path(root.path()), b"tampered").unwrap();
        assert!(PluginSet::prepare(&[decl("0.1.0")], &[root.path().into()]).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn latest_refresh_keeps_selected_snapshot_and_context_owns_the_handle() {
        let root = tempfile::tempdir().unwrap();
        let compile = |number| {
            let source = root.path().join(format!("{number}.c"));
            let binary = root.path().join(format!("{number}.so"));
            std::fs::write(
                &source,
                format!("int plugin_value(void) {{ return {number}; }}"),
            )
            .unwrap();
            let status = std::process::Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(source)
                .arg("-o")
                .arg(&binary)
                .status()
                .unwrap();
            assert!(status.success());
            std::fs::read(binary).unwrap()
        };
        let first = compile(11);
        let second = compile(22);
        publish(root.path(), "latest", "0.1.0", &first);
        publish(root.path(), "0.1.0", "0.1.0", &first);
        let old = PluginSet::prepare(&[decl("latest")], &[root.path().into()]).unwrap();
        let context = old.context();
        drop(old);
        publish(root.path(), "latest", "0.2.0", &second);
        let new = PluginSet::prepare(&[decl("latest")], &[root.path().into()]).unwrap();
        let exact = PluginSet::prepare(&[decl("0.1.0")], &[root.path().into()]).unwrap();
        let call = |context: RuntimeContext| unsafe {
            let function: unsafe extern "C" fn() -> i32 =
                std::mem::transmute(context.symbol("fixture-bridge", "plugin_value").unwrap());
            function()
        };
        // Old selection had not opened its library before refreshing latest.
        assert_eq!(call(context.clone()), 11);
        assert_eq!(call(new.context()), 22);
        assert_eq!(call(exact.context()), 11);
        assert!(context.symbol("undeclared", "plugin_value").is_err());
        assert!(context.symbol("fixture-bridge", "missing").is_err());
    }
}
