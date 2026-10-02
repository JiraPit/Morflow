//! Process-resident libraries keyed by immutable, verified snapshot paths.
//!
//! Native actions/plugins may own worker pools, thread-local destructors, or
//! returned storage whose lifetime exceeds a pipeline. Unloading their code
//! while those remain alive is unsafe. Version selection remains pipeline-local;
//! only the lifetime of each selected native module is process-wide.
use libloading::Library;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) fn library(path: &Path) -> Result<Arc<Library>, String> {
    static LIBRARIES: OnceLock<Mutex<HashMap<PathBuf, Arc<Library>>>> = OnceLock::new();
    let mut libraries = LIBRARIES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|e| e.to_string())?;
    if let Some(library) = libraries.get(path) {
        return Ok(library.clone());
    }
    let library = Arc::new(unsafe { Library::new(path) }.map_err(|e| e.to_string())?);
    libraries.insert(path.into(), library.clone());
    Ok(library)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn selected_modules_remain_resident_and_failed_loads_can_retry() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("fixture.so");
        assert!(library(&path).is_err());
        let source = root.path().join("fixture.c");
        std::fs::write(&source, "int native_value(void) { return 7; }").unwrap();
        assert!(std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(source)
            .arg("-o")
            .arg(&path)
            .status()
            .unwrap()
            .success());
        let first = library(&path).unwrap();
        assert!(Arc::ptr_eq(&first, &library(&path).unwrap()));
        let weak = Arc::downgrade(&first);
        drop(first);
        let retained = weak
            .upgrade()
            .expect("Worker threads and returned storage require resident code");
        let value = unsafe {
            retained
                .get::<unsafe extern "C" fn() -> i32>(b"native_value")
                .unwrap()()
        };
        assert_eq!(value, 7);
    }
}
