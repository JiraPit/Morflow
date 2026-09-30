//! Release discovery and checksum-verified installation. Runtime never calls this module.
use crate::artifact::*;
use semver::Version;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Read;
use std::path::Path;

pub trait Transport {
    fn get(&self, url: &str) -> Result<Vec<u8>, String>;
}
pub struct HttpTransport;
impl Transport for HttpTransport {
    fn get(&self, url: &str) -> Result<Vec<u8>, String> {
        let response = ureq::get(url)
            .set("User-Agent", "Morflow-CLI")
            .call()
            .map_err(|e| format!("{url}: {e}"))?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
}
#[derive(Debug, Clone, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}
#[derive(Debug, Clone, Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    assets: Vec<Asset>,
}
#[derive(Clone)]
pub struct PreparedRelease {
    pub catalog: ReleaseCatalog,
    urls: HashMap<String, String>,
}
pub struct ReleaseClient<T = HttpTransport> {
    pub repository: String,
    pub transport: T,
}
impl ReleaseClient<HttpTransport> {
    pub fn new(repository: String) -> Result<Self, String> {
        let parts: Vec<_> = repository.split('/').collect();
        if parts.len() != 2 {
            return Err("Repository must be owner/name".into());
        }
        component(parts[0])?;
        component(parts[1])?;
        Ok(Self {
            repository,
            transport: HttpTransport,
        })
    }
}
impl<T: Transport> ReleaseClient<T> {
    fn api(&self) -> String {
        format!("https://api.github.com/repos/{}", self.repository)
    }
    pub fn resolve(&self, pack: &str, requested: &str) -> Result<PreparedRelease, String> {
        component(pack)?;
        let requested = normalize_version(requested)?;
        let release = if requested == "latest" {
            let mut latest: Option<(Version, Release)> = None;
            for page in 1.. {
                let url = format!("{}/releases?per_page=100&page={page}", self.api());
                let releases: Vec<Release> = serde_json::from_slice(&self.transport.get(&url)?)
                    .map_err(|e| e.to_string())?;
                let count = releases.len();
                for release in releases {
                    if release.draft || release.prerelease {
                        continue;
                    }
                    if let Some(raw) = release
                        .tag_name
                        .strip_prefix(&format!("action_packs/{pack}/v"))
                    {
                        if let Ok(version) = Version::parse(raw) {
                            if version.pre.is_empty()
                                && latest.as_ref().is_none_or(|(v, _)| version > *v)
                            {
                                latest = Some((version, release));
                            }
                        }
                    }
                }
                if count < 100 {
                    break;
                }
            }
            latest
                .ok_or_else(|| format!("No stable released version of {pack}"))?
                .1
        } else {
            let url = format!(
                "{}/releases/tags/action_packs%2F{}%2Fv{}",
                self.api(),
                pack,
                requested.replace('+', "%2B")
            );
            let release: Release =
                serde_json::from_slice(&self.transport.get(&url)?).map_err(|e| e.to_string())?;
            if release.draft || release.tag_name != format!("action_packs/{pack}/v{requested}") {
                return Err(format!("Release does not match {pack}/{requested}"));
            }
            release
        };
        let concrete = release
            .tag_name
            .strip_prefix(&format!("action_packs/{pack}/v"))
            .ok_or("Invalid release tag")?
            .to_owned();
        let urls: HashMap<_, _> = release
            .assets
            .into_iter()
            .map(|a| (a.name, a.browser_download_url))
            .collect();
        let checksum_url = urls
            .get("checksums.txt")
            .ok_or_else(|| format!("Release {pack}/{concrete} has no checksums.txt"))?;
        let checksums = parse_checksums(&self.transport.get(checksum_url)?)?;
        Ok(PreparedRelease {
            catalog: ReleaseCatalog {
                pack: pack.into(),
                requested_version: requested,
                concrete_version: concrete,
                repository: self.repository.clone(),
                checksums,
            },
            urls,
        })
    }
    pub fn install(
        &self,
        root: &Path,
        identity: &ActionIdentity,
        release: &PreparedRelease,
        force: bool,
    ) -> Result<bool, String> {
        let catalog = &release.catalog;
        if normalize_version(&catalog.concrete_version)? == "latest"
            || (identity.version != "latest" && identity.version != catalog.concrete_version)
        {
            return Err("Concrete release does not match the requested version".into());
        }
        if identity.pack != catalog.pack
            || identity.version != catalog.requested_version
            || catalog.repository != self.repository
        {
            return Err("Installation identity/release mismatch".into());
        }
        let remote_id = ActionIdentity::for_platform(
            &identity.pack,
            &catalog.concrete_version,
            &identity.action,
            &identity.platform,
        )?;
        let filename = remote_id.filename();
        let expected = catalog.checksums.get(&filename).ok_or_else(|| {
            format!(
                "Requested action {identity} is absent from release {}",
                catalog.concrete_version
            )
        })?;
        let url = release
            .urls
            .get(&filename)
            .ok_or_else(|| format!("Release asset {filename} is missing"))?;
        let _maintenance = CacheGuard::shared(root, "maintenance")?;
        let _guard = CacheGuard::acquire(root, &identity.to_string())?;
        let path = identity.path(root);
        let cached = if !force {
            fs::read(&path)
                .ok()
                .filter(|bytes| digest(bytes) == *expected)
        } else {
            None
        };
        let downloaded = cached.is_none();
        let bytes = match cached {
            Some(bytes) => bytes,
            None => self.transport.get(url)?,
        };
        if digest(&bytes) != *expected {
            return Err(format!(
                "Published checksum mismatch for {identity}; cached binary was preserved"
            ));
        }
        // Download and verify documentation before changing any artifact files.
        let spec_name = format!("{}_SPEC.md", identity.action);
        let spec = if let Some(hash) = catalog.checksums.get(&spec_name) {
            let spec_path = spec_path(&path);
            let data = fs::read(&spec_path)
                .ok()
                .filter(|bytes| !force && digest(bytes) == *hash);
            let data = match data {
                Some(data) => data,
                None => self.transport.get(
                    release
                        .urls
                        .get(&spec_name)
                        .ok_or("Missing specification asset")?,
                )?,
            };
            if digest(&data) != *hash {
                return Err(format!("Specification checksum mismatch for {identity}"));
            }
            Some(data)
        } else {
            None
        };
        let receipt = ArtifactReceipt {
            identity: identity.clone(),
            concrete_version: catalog.concrete_version.clone(),
            repository: self.repository.clone(),
            sha256: expected.clone(),
        };
        // Prepare the immutable load location before publishing the mutable alias.
        snapshot(root, identity, &bytes, expected)?;
        if downloaded {
            atomic_write(&path, &bytes)?;
        }
        if let Some(spec) = spec {
            atomic_write(&spec_path(&path), &spec)?;
        }
        atomic_write(
            &receipt_path(&path),
            &serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
        )?;
        catalog.write(root)?;
        Ok(downloaded)
    }
    pub fn spec(&self, release: &PreparedRelease, action: &str) -> Result<Vec<u8>, String> {
        component(action)?;
        let name = format!("{action}_SPEC.md");
        let expected = release
            .catalog
            .checksums
            .get(&name)
            .ok_or("Specification has no published checksum")?;
        let bytes = self.transport.get(
            release
                .urls
                .get(&name)
                .ok_or("Specification is absent from release")?,
        )?;
        if digest(&bytes) != *expected {
            return Err("Specification checksum mismatch".into());
        }
        Ok(bytes)
    }
}
pub fn spec_path(binary: &Path) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("{}.SPEC.md", binary.display()))
}
pub fn parse_checksums(bytes: &[u8]) -> Result<BTreeMap<String, String>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut result = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (hash, name) = line
            .split_once(char::is_whitespace)
            .ok_or("Malformed checksum line")?;
        let name = name.trim().trim_start_matches('*');
        component(name)?;
        if hash.len() != 64
            || !hash.bytes().all(|b| b.is_ascii_hexdigit())
            || result
                .insert(name.into(), hash.to_ascii_lowercase())
                .is_some()
        {
            return Err(format!("Invalid or duplicate checksum for {name}"));
        }
    }
    if result.is_empty() {
        return Err("Empty release checksum manifest".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    #[derive(Default, Clone)]
    struct Fake {
        data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        requests: Arc<Mutex<Vec<String>>>,
    }
    impl Transport for Fake {
        fn get(&self, url: &str) -> Result<Vec<u8>, String> {
            self.requests.lock().unwrap().push(url.into());
            self.data
                .lock()
                .unwrap()
                .get(url)
                .cloned()
                .ok_or_else(|| format!("Missing test response: {url}"))
        }
    }
    fn fixture(fake: &Fake, version: &str, bytes: &[u8], requested: &str) -> PreparedRelease {
        let id = ActionIdentity::new("base", version, "identity").unwrap();
        let url = format!("test://{version}/binary");
        fake.data
            .lock()
            .unwrap()
            .insert(url.clone(), bytes.to_vec());
        PreparedRelease {
            catalog: ReleaseCatalog {
                pack: "base".into(),
                requested_version: requested.into(),
                concrete_version: version.into(),
                repository: "owner/repo".into(),
                checksums: BTreeMap::from([(id.filename(), digest(bytes))]),
            },
            urls: HashMap::from([(id.filename(), url)]),
        }
    }
    #[test]
    fn latest_refresh_exact_coexistence_and_failed_download_preservation() {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::default();
        let client = ReleaseClient {
            repository: "owner/repo".into(),
            transport: fake.clone(),
        };
        let latest = ActionIdentity::new("base", "latest", "identity").unwrap();
        let old = fixture(&fake, "0.1.0", b"old", "latest");
        assert!(client.install(root.path(), &latest, &old, false).unwrap());
        assert!(!client.install(root.path(), &latest, &old, false).unwrap());
        assert_eq!(fake.requests.lock().unwrap().len(), 1);
        let exact = ActionIdentity::new("base", "0.1.0", "identity").unwrap();
        client
            .install(
                root.path(),
                &exact,
                &fixture(&fake, "0.1.0", b"old", "0.1.0"),
                false,
            )
            .unwrap();
        let new = fixture(&fake, "0.2.0", b"new", "latest");
        client.install(root.path(), &latest, &new, false).unwrap();
        assert_eq!(fs::read(exact.path(root.path())).unwrap(), b"old");
        assert_eq!(
            read_verified(root.path(), &latest)
                .unwrap()
                .0
                .concrete_version,
            "0.2.0"
        );
        let same = fixture(&fake, "0.3.0", b"new", "latest");
        assert!(!client.install(root.path(), &latest, &same, false).unwrap());
        assert_eq!(
            read_verified(root.path(), &latest)
                .unwrap()
                .0
                .concrete_version,
            "0.3.0"
        );
        let broken = fixture(&fake, "0.4.0", b"expected", "latest");
        fake.data
            .lock()
            .unwrap()
            .insert("test://0.4.0/binary".into(), b"wrong".to_vec());
        assert!(client
            .install(root.path(), &latest, &broken, false)
            .is_err());
        assert_eq!(
            read_verified(root.path(), &latest)
                .unwrap()
                .0
                .concrete_version,
            "0.3.0"
        );
        fake.data.lock().unwrap().remove("test://0.4.0/binary");
        assert!(client
            .install(root.path(), &latest, &broken, false)
            .is_err());
        assert_eq!(fs::read(latest.path(root.path())).unwrap(), b"new");
    }
    #[test]
    fn latest_uses_semver_and_follows_pagination() {
        let fake = Fake::default();
        let mut first: Vec<serde_json::Value> = (0..98)
            .map(|_| serde_json::json!({"tag_name":"other/v99.0.0","assets":[]}))
            .collect();
        first.push(serde_json::json!({"tag_name":"action_packs/base/v0.9.0","assets":[]}));
        first.push(serde_json::json!({"tag_name":"action_packs/base/v9.0.0","prerelease":true,"assets":[]}));
        let asset = ActionIdentity::new("base", "0.10.0", "identity")
            .unwrap()
            .filename();
        let second = serde_json::json!([
            {"tag_name":"action_packs/base/v10.0.0","draft":true,"assets":[]},
            {"tag_name":"action_packs/base/v0.10.0","assets":[{"name":"checksums.txt","browser_download_url":"test://checksums"}]}
        ]);
        fake.data.lock().unwrap().extend([
            (
                "https://api.github.com/repos/owner/repo/releases?per_page=100&page=1".into(),
                serde_json::to_vec(&first).unwrap(),
            ),
            (
                "https://api.github.com/repos/owner/repo/releases?per_page=100&page=2".into(),
                serde_json::to_vec(&second).unwrap(),
            ),
            (
                "test://checksums".into(),
                format!("{}  {asset}\n", digest(b"binary")).into_bytes(),
            ),
        ]);
        let client = ReleaseClient {
            repository: "owner/repo".into(),
            transport: fake.clone(),
        };
        let release = client.resolve("base", "latest").unwrap();
        assert_eq!(release.catalog.concrete_version, "0.10.0");
        assert!(fake
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|u| u.ends_with("page=2")));
    }
    #[test]
    fn exact_resolution_queries_only_the_requested_release() {
        let fake = Fake::default();
        let filename = ActionIdentity::new("base", "0.2.0", "identity")
            .unwrap()
            .filename();
        fake.data.lock().unwrap().extend([
            ("https://api.github.com/repos/owner/repo/releases/tags/action_packs%2Fbase%2Fv0.2.0".into(), serde_json::to_vec(&serde_json::json!({
                "tag_name": "action_packs/base/v0.2.0", "assets": [{"name": "checksums.txt", "browser_download_url": "test://exact-checksums"}]
            })).unwrap()),
            ("test://exact-checksums".into(), format!("{}  {filename}\n", digest(b"exact")).into_bytes())
        ]);
        let client = ReleaseClient {
            repository: "owner/repo".into(),
            transport: fake.clone(),
        };
        assert_eq!(
            client
                .resolve("base", "0.2.0")
                .unwrap()
                .catalog
                .concrete_version,
            "0.2.0"
        );
        assert!(!fake
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|u| u.contains("per_page")));
        assert!(client.resolve("base", "0.1.0").is_err());
    }

    #[test]
    fn missing_assets_and_malformed_manifests_fail() {
        assert!(parse_checksums(b"not-a-hash  a.so\n").is_err());
        assert!(parse_checksums(format!("{}  ../a.so\n", digest(b"x")).as_bytes()).is_err());
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::default();
        let client = ReleaseClient {
            repository: "owner/repo".into(),
            transport: fake.clone(),
        };
        let id = ActionIdentity::new("base", "0.2.0", "missing").unwrap();
        assert!(client
            .install(
                root.path(),
                &id,
                &fixture(&fake, "0.2.0", b"binary", "0.2.0"),
                false
            )
            .is_err());
        assert!(!id.path(root.path()).exists());
    }
    #[test]
    fn concurrent_writers_and_readers_observe_consistent_pairs() {
        let root = tempfile::tempdir().unwrap();
        let fake = Fake::default();
        let client = Arc::new(ReleaseClient {
            repository: "owner/repo".into(),
            transport: fake.clone(),
        });
        let id = ActionIdentity::new("base", "latest", "identity").unwrap();
        let old = fixture(&fake, "0.1.0", b"first", "latest");
        let next = fixture(&fake, "0.2.0", b"second", "latest");
        client.install(root.path(), &id, &old, false).unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..20 {
                    client.install(root.path(), &id, &next, true).unwrap();
                    client.install(root.path(), &id, &old, true).unwrap();
                }
            });
            scope.spawn(|| {
                for _ in 0..40 {
                    let _guard = CacheGuard::acquire(root.path(), &id.to_string()).unwrap();
                    let (r, bytes) = read_verified(root.path(), &id).unwrap();
                    r.validate(&id, &bytes).unwrap();
                }
            });
        });
    }
}
