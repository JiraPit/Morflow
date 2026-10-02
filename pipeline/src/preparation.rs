//! Resolve all action and plugin dependencies before modifying either cache.
use crate::artifact::ActionIdentity;
use crate::engine::collect_action_names;
use crate::plugins::declarations;
use crate::release::{PreparedRelease, ReleaseClient, Transport};
use crate::resolver::ActionResolver;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug)]
pub struct Preparation {
    pub actions: usize,
    pub plugins: usize,
}
fn release<'a, T: Transport>(
    client: &ReleaseClient<T>,
    releases: &'a mut HashMap<(String, String), PreparedRelease>,
    pack: &str,
    version: &str,
) -> Result<&'a PreparedRelease, String> {
    let key = (pack.to_owned(), version.to_owned());
    if !releases.contains_key(&key) {
        releases.insert(key.clone(), client.resolve(pack, version)?);
    }
    Ok(&releases[&key])
}
pub fn prepare<T: Transport>(
    ast: &parser::ast::Pipeline,
    client: &ReleaseClient<T>,
    action_root: &Path,
    plugin_root: &Path,
    force: bool,
) -> Result<Preparation, String> {
    crate::validator::validate_pipeline(ast).map_err(|e| e.to_string())?;
    let plugins = declarations(&ast.plugins)?;
    let resolver = ActionResolver::from_imports(&ast.imports)?;
    let mut releases = HashMap::new();
    let mut identities = HashSet::new();
    for name in collect_action_names(&ast.statements) {
        let id = resolver.resolve(&name, |p, v| {
            Ok(release(client, &mut releases, p, v)?
                .catalog
                .actions(crate::cli::get_host_platform().0))
        })?;
        release(client, &mut releases, &id.pack, &id.version)?;
        identities.insert(id);
    }
    let mut identities: Vec<ActionIdentity> = identities.into_iter().collect();
    identities.sort_by_key(ToString::to_string);
    let mut plugin_releases = HashMap::new();
    for (name, id) in &plugins {
        plugin_releases.insert(name.clone(), client.resolve_plugin(name, &id.version)?);
    }
    for id in &identities {
        for req in client.action_requirements(
            &releases[&(id.pack.clone(), id.version.clone())],
            &id.action,
        )? {
            let selected = plugin_releases.get(&req.name).ok_or_else(|| {
                format!(
                    "Required plugin '{}' is undeclared; add plugin {}/<version>",
                    req.name, req.name
                )
            })?;
            if !semver::VersionReq::parse(&req.version)
                .map_err(|e| e.to_string())?
                .matches(
                    &semver::Version::parse(&selected.catalog.concrete_version)
                        .map_err(|e| e.to_string())?,
                )
            {
                return Err(format!(
                    "Plugin {} requires {}, but selected {}",
                    req.name, req.version, selected.catalog.concrete_version
                ));
            }
        }
    }
    for (name, id) in &plugins {
        client.install_plugin(plugin_root, id, &plugin_releases[name], force)?;
    }
    for id in &identities {
        client.install(
            action_root,
            id,
            &releases[&(id.pack.clone(), id.version.clone())],
            force,
        )?;
    }
    for release in releases.values() {
        release.catalog.write(action_root)?;
    }
    Ok(Preparation {
        actions: identities.len(),
        plugins: plugins.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{digest, read_verified, receipt_path};
    use crate::plugins;
    use crate::plugins::PluginIdentity;
    use std::sync::{Arc, Mutex};
    #[derive(Clone, Default)]
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
                .ok_or_else(|| format!("Missing fixture {url}"))
        }
    }
    fn fixture(
        fake: &Fake,
        namespace: &str,
        name: &str,
        version: &str,
        filename: &str,
        bytes: &[u8],
        metadata: Option<&[u8]>,
    ) {
        let base = format!("test://{namespace}/{name}/{version}");
        let mut assets = vec![
            serde_json::json!({"name":filename,"browser_download_url":format!("{base}/binary")}),
            serde_json::json!({"name":"checksums.txt","browser_download_url":format!("{base}/checksums")}),
        ];
        let mut checksums = format!("{}  {filename}\n", digest(bytes));
        let mut data = fake.data.lock().unwrap();
        data.insert(format!("{base}/binary"), bytes.to_vec());
        if let Some(metadata) = metadata {
            let name = "resize_METADATA.json";
            assets.push(
                serde_json::json!({"name":name,"browser_download_url":format!("{base}/metadata")}),
            );
            checksums += &format!("{}  {name}\n", digest(metadata));
            data.insert(format!("{base}/metadata"), metadata.to_vec());
        }
        data.insert(format!("{base}/checksums"), checksums.into_bytes());
        data.insert(format!("https://api.github.com/repos/test/repo/releases/tags/{namespace}%2F{name}%2Fv{version}"),serde_json::to_vec(&serde_json::json!({"tag_name":format!("{namespace}/{name}/v{version}"),"assets":assets})).unwrap());
    }
    fn setup() -> (Fake, ReleaseClient<Fake>, ActionIdentity, PluginIdentity) {
        let fake = Fake::default();
        let action = ActionIdentity::new("image_opencv", "0.1.0", "resize").unwrap();
        let plugin = PluginIdentity::new("opencv-bridge", "0.1.0").unwrap();
        fixture(
            &fake,
            "action_packs",
            "image_opencv",
            "0.1.0",
            &action.filename(),
            b"action",
            Some(br#"{"plugins":[{"name":"opencv-bridge","version":"^0.1.0"}]}"#),
        );
        fixture(
            &fake,
            "plugins",
            "opencv-bridge",
            "0.1.0",
            &plugin.filename(),
            b"plugin",
            None,
        );
        let client = ReleaseClient {
            repository: "test/repo".into(),
            transport: fake.clone(),
        };
        (fake, client, action, plugin)
    }
    fn source(plugin: &str) -> parser::ast::Pipeline {
        parser::parse(&format!("{plugin}\nfrom image_opencv/0.1.0 import resize\naccept Tensor[2,3] $x\n$x >> resize(3,2) >> emit\n")).unwrap()
    }
    #[test]
    fn prep_installs_declared_plugins_and_actions_without_redundant_binary_downloads() {
        let (fake, client, action, plugin) = setup();
        let actions = tempfile::tempdir().unwrap();
        let plugins = tempfile::tempdir().unwrap();
        let ast = source("plugin opencv-bridge/0.1.0");
        let result = prepare(&ast, &client, actions.path(), plugins.path(), false).unwrap();
        assert_eq!((result.actions, result.plugins), (1, 1));
        assert_eq!(
            read_verified(actions.path(), &action).unwrap().0.plugins[0].name,
            "opencv-bridge"
        );
        assert_eq!(
            plugins::read_verified(plugins.path(), &plugin).unwrap().1,
            b"plugin"
        );
        prepare(&ast, &client, actions.path(), plugins.path(), false).unwrap();
        assert_eq!(
            fake.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|u| u.ends_with("/binary"))
                .count(),
            2
        );
    }
    #[test]
    fn prep_rejects_undeclared_incompatible_and_missing_plugins_before_installing() {
        let (fake, client, action, _) = setup();
        let actions = tempfile::tempdir().unwrap();
        let plugins = tempfile::tempdir().unwrap();
        assert!(
            prepare(&source(""), &client, actions.path(), plugins.path(), false)
                .unwrap_err()
                .contains("undeclared")
        );
        assert!(!action.path(actions.path()).exists());
        let id = PluginIdentity::new("opencv-bridge", "0.2.0").unwrap();
        fixture(
            &fake,
            "plugins",
            "opencv-bridge",
            "0.2.0",
            &id.filename(),
            b"next",
            None,
        );
        assert!(prepare(
            &source("plugin opencv-bridge/0.2.0"),
            &client,
            actions.path(),
            plugins.path(),
            false
        )
        .unwrap_err()
        .contains("requires"));
        assert!(prepare(
            &source("plugin opencv-bridge/9.0.0"),
            &client,
            actions.path(),
            plugins.path(),
            false
        )
        .is_err());
        assert!(!id.path(plugins.path()).exists());
        assert!(!receipt_path(&action.path(actions.path())).exists());
        assert!(fake
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|u| !u.ends_with("/binary")));
    }
    #[test]
    fn failed_plugin_download_preserves_verified_pair_and_prep_fails() {
        let (fake, client, _, id) = setup();
        let actions = tempfile::tempdir().unwrap();
        let plugins = tempfile::tempdir().unwrap();
        let ast = source("plugin opencv-bridge/0.1.0");
        prepare(&ast, &client, actions.path(), plugins.path(), false).unwrap();
        fake.data.lock().unwrap().insert(
            "test://plugins/opencv-bridge/0.1.0/binary".into(),
            b"bad checksum".to_vec(),
        );
        assert!(prepare(&ast, &client, actions.path(), plugins.path(), true).is_err());
        assert_eq!(
            plugins::read_verified(plugins.path(), &id).unwrap().1,
            b"plugin"
        );
        fake.data
            .lock()
            .unwrap()
            .remove("test://plugins/opencv-bridge/0.1.0/binary");
        assert!(prepare(&ast, &client, actions.path(), plugins.path(), true).is_err());
        assert_eq!(
            plugins::read_verified(plugins.path(), &id).unwrap().1,
            b"plugin"
        );
    }
    #[test]
    fn metadata_is_required_and_checksum_verified_before_any_installation() {
        let (fake, client, action, _) = setup();
        let actions = tempfile::tempdir().unwrap();
        let plugins = tempfile::tempdir().unwrap();
        let ast = source("plugin opencv-bridge/0.1.0");
        fake.data.lock().unwrap().insert(
            "test://action_packs/image_opencv/0.1.0/metadata".into(),
            b"modified".to_vec(),
        );
        assert!(
            prepare(&ast, &client, actions.path(), plugins.path(), false)
                .unwrap_err()
                .contains("metadata checksum mismatch")
        );
        fake.data.lock().unwrap().insert(
            "test://action_packs/image_opencv/0.1.0/checksums".into(),
            format!("{}  {}\n", digest(b"action"), action.filename()).into_bytes(),
        );
        assert!(
            prepare(&ast, &client, actions.path(), plugins.path(), false)
                .unwrap_err()
                .contains("missing required checksummed metadata")
        );
        assert!(fake
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|url| !url.ends_with("/binary")));
        assert!(!action.path(actions.path()).exists());
    }
}
