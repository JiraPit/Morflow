//! Engine-provided runtime access to pipeline-scoped plugins.
use crate::{RString, RVec, StableAbi};
use abi_stable::std_types::RArc;
use abi_stable::std_types::RResult;

#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct PluginRequirement {
    pub name: RString,
    /// A semantic version requirement, e.g. ^0.1.0.
    pub version: RString,
}
pub type GetPluginsFn = extern "C" fn() -> RVec<PluginRequirement>;

/// Only the engine creates this context. Its owner remains alive throughout
/// synchronous process calls, including parallel action invocations.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct RuntimeContext {
    owner: RArc<PluginOwner>,
    resolve: extern "C" fn(usize, RString, RString) -> RResult<usize, RString>,
}
#[repr(C)]
#[derive(StableAbi, Debug)]
struct PluginOwner {
    pointer: usize,
    release: extern "C" fn(usize),
}
impl Drop for PluginOwner {
    fn drop(&mut self) {
        (self.release)(self.pointer);
    }
}
impl RuntimeContext {
    /// Construct an owning runtime context.
    ///
    /// # Safety
    /// `pointer` must remain valid until `release` is called exactly once. The
    /// callbacks must be thread-safe, and must never unwind across the ABI.
    pub unsafe fn new(
        pointer: usize,
        release: extern "C" fn(usize),
        resolve: extern "C" fn(usize, RString, RString) -> RResult<usize, RString>,
    ) -> Self {
        Self {
            owner: RArc::new(PluginOwner { pointer, release }),
            resolve,
        }
    }

    /// Return a symbol from the declared, verified plugin. Loading is deferred
    /// until this call; callers must use the plugin's documented function type.
    pub fn symbol(&self, plugin: &str, symbol: &str) -> Result<usize, RString> {
        (self.resolve)(self.owner.pointer, plugin.into(), symbol.into()).into_result()
    }
}
