//! Versioned artifacts, checksum receipts, and coordinated cache access.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActionIdentity {
    pub pack: String,
    pub version: String,
    pub action: String,
    pub platform: String,
}

pub fn component(value: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-+.".contains(&b))
        || value == "."
        || value == ".."
    {
        return Err(format!("Invalid artifact name component '{value}'"));
    }
    Ok(())
}

pub fn normalize_version(version: &str) -> Result<String, String> {
    if version == "latest" {
        return Ok(version.into());
    }
    let version = version.strip_prefix('v').unwrap_or(version);
    semver::Version::parse(version).map_err(|e| {
        format!("Expected an exact semantic version or latest, got '{version}': {e}")
    })?;
    Ok(version.into())
}

impl ActionIdentity {
    pub fn new(pack: &str, version: &str, action: &str) -> Result<Self, String> {
        Self::for_platform(pack, version, action, crate::cli::get_host_platform().0)
    }
    pub fn for_platform(
        pack: &str,
        version: &str,
        action: &str,
        platform: &str,
    ) -> Result<Self, String> {
        component(pack)?;
        component(action)?;
        component(platform)?;
        Ok(Self {
            pack: pack.into(),
            version: normalize_version(version)?,
            action: action.into(),
            platform: platform.into(),
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
            "{}_action-{}-{}.{}",
            self.action, self.version, self.platform, ext
        )
    }
    pub fn path(&self, root: &Path) -> PathBuf {
        root.join(&self.pack).join(self.filename())
    }
}
impl std::fmt::Display for ActionIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{} ({})",
            self.pack, self.version, self.action, self.platform
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactReceipt {
    pub plugins: Vec<crate::plugins::Requirement>,
    #[serde(flatten)]
    pub identity: ActionIdentity,
    pub concrete_version: String,
    pub repository: String,
    pub sha256: String,
}
impl ArtifactReceipt {
    pub fn validate(&self, identity: &ActionIdentity, bytes: &[u8]) -> Result<(), String> {
        for requirement in &self.plugins {
            requirement.validate()?;
        }
        let canonical = ActionIdentity::for_platform(
            &identity.pack,
            &identity.version,
            &identity.action,
            &identity.platform,
        )?;
        if canonical != *identity {
            return Err("Noncanonical artifact identity".into());
        }
        if &self.identity != identity
            || self.concrete_version == "latest"
            || normalize_version(&self.concrete_version)? != self.concrete_version
            || (identity.version != "latest" && identity.version != self.concrete_version)
        {
            return Err(format!(
                "Receipt identity does not match requested {identity}"
            ));
        }
        if self.repository.is_empty() || self.sha256 != digest(bytes) {
            return Err(format!(
                "Checksum or provenance verification failed for {identity}"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseCatalog {
    pub pack: String,
    pub requested_version: String,
    pub concrete_version: String,
    pub repository: String,
    pub checksums: BTreeMap<String, String>,
}
impl ReleaseCatalog {
    pub fn actions(&self, platform: &str) -> Vec<String> {
        let suffix = format!(
            "_action-{}-{}.{}",
            self.concrete_version,
            platform,
            if platform.starts_with("windows-") {
                "dll"
            } else if platform.starts_with("darwin-") {
                "dylib"
            } else {
                "so"
            }
        );
        self.checksums
            .keys()
            .filter_map(|n| n.strip_suffix(&suffix).map(str::to_owned))
            .collect()
    }
    pub fn path(root: &Path, pack: &str, version: &str) -> PathBuf {
        root.join(pack)
            .join(".catalogs")
            .join(format!("{version}.json"))
    }
    pub fn read(root: &Path, pack: &str, version: &str) -> Result<Self, String> {
        component(pack)?;
        let version = normalize_version(version)?;
        let _maintenance = CacheGuard::shared(root, "maintenance")?;
        let path = Self::path(root, pack, &version);
        let _guard = CacheGuard::acquire(root, &format!("catalog-{pack}-{version}"))?;
        let result: Self = read_json(&path)?;
        if result.pack != pack
            || result.requested_version != version
            || result.concrete_version == "latest"
            || normalize_version(&result.concrete_version)? != result.concrete_version
            || (version != "latest" && result.concrete_version != version)
        {
            return Err(format!("Invalid catalog for {pack}/{version}"));
        }
        Ok(result)
    }
    pub fn write(&self, root: &Path) -> Result<(), String> {
        component(&self.pack)?;
        normalize_version(&self.requested_version)?;
        if self.requested_version == "latest" {
            let mut concrete = self.clone();
            concrete.requested_version = self.concrete_version.clone();
            if concrete.requested_version == "latest" {
                return Err("A catalog must identify a concrete release".into());
            }
            concrete.write(root)?;
        }
        let _maintenance = CacheGuard::shared(root, "maintenance")?;
        let _guard = CacheGuard::acquire(
            root,
            &format!("catalog-{}-{}", self.pack, self.requested_version),
        )?;
        atomic_write(
            &Self::path(root, &self.pack, &self.requested_version),
            &serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?,
        )
    }
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn receipt_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", path.display()))
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    serde_json::from_slice(
        &fs::read(path)
            .map_err(|e| format!("{}: {e}; run morflow prep or install", path.display()))?,
    )
    .map_err(|e| format!("Invalid metadata at {}: {e}", path.display()))
}

/// OS locks release automatically on drop, including after a crashed writer.
pub struct CacheGuard(File);
impl CacheGuard {
    pub fn acquire(root: &Path, key: &str) -> Result<Self, String> {
        Self::open(root, key, false)
    }
    pub fn shared(root: &Path, key: &str) -> Result<Self, String> {
        Self::open(root, key, true)
    }
    fn open(root: &Path, key: &str, shared: bool) -> Result<Self, String> {
        let dir = root.join(".locks");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(format!("{}.lck", digest(key.as_bytes()))))
            .map_err(|e| e.to_string())?;
        if shared {
            file.lock_shared().map_err(|e| e.to_string())?;
        } else {
            file.lock().map_err(|e| e.to_string())?;
        }
        Ok(Self(file))
    }
}
impl Drop for CacheGuard {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("Missing artifact directory")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temp.write_all(bytes).map_err(|e| e.to_string())?;
    temp.as_file().sync_all().map_err(|e| e.to_string())?;
    temp.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}

/// Caller holds the artifact guard. Snapshot paths are never replaced.
pub fn snapshot(
    root: &Path,
    identity: &ActionIdentity,
    bytes: &[u8],
    hash: &str,
) -> Result<PathBuf, String> {
    let path = root.join(".objects").join(hash).join(identity.filename());
    let _guard = CacheGuard::acquire(root, &format!("snapshot-{hash}-{}", identity.filename()))?;
    if path.exists() {
        if digest(&fs::read(&path).map_err(|e| e.to_string())?) != hash {
            return Err(format!("Corrupt immutable snapshot {}", path.display()));
        }
    } else {
        atomic_write(&path, bytes)?;
    }
    fs::canonicalize(path).map_err(|e| e.to_string())
}

pub fn read_verified(
    root: &Path,
    identity: &ActionIdentity,
) -> Result<(ArtifactReceipt, Vec<u8>), String> {
    let path = identity.path(root);
    let receipt: ArtifactReceipt = read_json(&receipt_path(&path))?;
    let bytes = fs::read(&path)
        .map_err(|e| format!("Cannot read {identity}: {e}; run morflow prep or install"))?;
    receipt.validate(identity, &bytes)?;
    Ok((receipt, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_identity_hash_and_exact_version() {
        let id = ActionIdentity::new("base", "0.2.0", "identity").unwrap();
        let mut r = ArtifactReceipt {
            plugins: Vec::new(),
            identity: id.clone(),
            concrete_version: "0.2.0".into(),
            repository: "owner/repo".into(),
            sha256: digest(b"binary"),
        };
        assert!(r.validate(&id, b"binary").is_ok());
        assert!(r.validate(&id, b"corrupted").is_err());
        r.concrete_version = "0.1.0".into();
        assert!(r.validate(&id, b"binary").is_err());
        r.concrete_version = "0.2.0".into();
        r.identity.pack = "other".into();
        assert!(r.validate(&id, b"binary").is_err());
    }
    #[test]
    fn immutable_snapshots_are_content_addressed() {
        let root = tempfile::tempdir().unwrap();
        let id = ActionIdentity::new("base", "latest", "identity").unwrap();
        let first = snapshot(root.path(), &id, b"first", &digest(b"first")).unwrap();
        let next = snapshot(root.path(), &id, b"next", &digest(b"next")).unwrap();
        assert_ne!(first, next);
        assert_eq!(fs::read(&first).unwrap(), b"first");
        fs::write(first, b"broken").unwrap();
        assert!(snapshot(root.path(), &id, b"first", &digest(b"first")).is_err());
    }
}
