//! Resolve all action and plugin dependencies before modifying either cache.
use crate::artifact::ActionIdentity;
use crate::engine::collect_action_names;
use crate::plugins::declarations;
use crate::release::{PreparedRelease, ReleaseClient, Transport};
use crate::resolver::ActionResolver;
use std::collections::{BTreeMap, HashMap, HashSet};
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
                .actions(crate::platform::host_platform().0))
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

/// Platforms an app can be built for, with their library file extension. A
/// pack an app bundles needs a pack library for each.
pub const APP_PLATFORMS: [(&str, &str); 9] = [
    ("linux-x86_64", "so"),
    ("linux-aarch64", "so"),
    ("darwin-arm64", "dylib"),
    // macOS release builds are universal, so Intel Macs too.
    ("darwin-x86_64", "dylib"),
    ("windows-x86_64", "dll"),
    ("android-aarch64", "so"),
    ("android-x86_64", "so"),
    ("ios-arm64", "dylib"),
    ("ios-arm64-sim", "dylib"),
];

/// A pack library's file name in the pack's release (see scripts/native_targets.py).
pub fn pack_library_asset(pack: &str, version: &str, platform: &str, extension: &str) -> String {
    format!("morflow_pack_{pack}-{version}-{platform}.{extension}")
}

/// The release of `pack` `version` (which may be `latest`), checked to have
/// a pack library for every app platform.
pub fn app_pack<T: Transport>(
    client: &ReleaseClient<T>,
    pack: &str,
    version: &str,
) -> Result<PreparedRelease, String> {
    let release = client.resolve(pack, version)?;
    check_app_pack(&release, pack, &release.catalog.concrete_version)?;
    Ok(release)
}

/// An error unless `release` has a pack library for every app platform.
fn check_app_pack(release: &PreparedRelease, pack: &str, version: &str) -> Result<(), String> {
    let checksums = &release.catalog.checksums;
    let missing: Vec<&str> = APP_PLATFORMS
        .iter()
        .filter(|(platform, ext)| {
            !checksums.contains_key(&pack_library_asset(pack, version, platform, ext))
        })
        .map(|(platform, _)| *platform)
        .collect();
    if missing.len() == APP_PLATFORMS.len() {
        return Err(format!(
            "{pack} {version} has no pack library, so apps cannot bundle it \
             (its actions need a plugin, or it was released before pack libraries)"
        ));
    }
    if !missing.is_empty() {
        return Err(format!(
            "{pack} {version} has no pack library for {} yet",
            missing.join(", ")
        ));
    }
    Ok(())
}

/// The packs an app must bundle to run `pipelines`: pack name -> exact
/// version, checked to have a pack library for every app platform. An app
/// bundles one version of each pack, which also serves `latest`, so a pack
/// one pipeline pins is the version every pipeline gets.
pub fn app_packs<T: Transport>(
    pipelines: &[parser::ast::Pipeline],
    client: &ReleaseClient<T>,
) -> Result<BTreeMap<String, String>, String> {
    let mut releases = HashMap::new();
    let mut exact: BTreeMap<String, String> = BTreeMap::new();
    let mut latest: HashSet<String> = HashSet::new();
    for ast in pipelines {
        crate::validator::validate_pipeline(ast).map_err(|e| e.to_string())?;
        if let Some(plugin) = ast.plugins.first() {
            return Err(format!(
                "The pipeline uses plugin {}; apps cannot bundle plugins or the actions that need them. \
                 Use a plugin-free pack instead.",
                plugin.name
            ));
        }
        let resolver = ActionResolver::from_imports(&ast.imports)?;
        for name in collect_action_names(&ast.statements) {
            let id = resolver.resolve(&name, |p, v| {
                Ok(release(client, &mut releases, p, v)?
                    .catalog
                    .actions(crate::platform::host_platform().0))
            })?;
            if id.version == "latest" {
                latest.insert(id.pack);
            } else if let Some(other) = exact.get(&id.pack).filter(|v| **v != id.version) {
                return Err(format!(
                    "The pipelines need {} {} and {} {}, but an app bundles one version of each pack",
                    id.pack, other, id.pack, id.version
                ));
            } else {
                exact.insert(id.pack, id.version);
            }
        }
    }
    let mut packs = exact;
    for pack in latest {
        if let std::collections::btree_map::Entry::Vacant(entry) = packs.entry(pack) {
            let concrete = release(client, &mut releases, entry.key(), "latest")?
                .catalog
                .concrete_version
                .clone();
            entry.insert(concrete);
        }
    }
    for (pack, version) in &packs {
        check_app_pack(
            release(client, &mut releases, pack, version)?,
            pack,
            version,
        )?;
    }
    Ok(packs)
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

    /// Publishes `pack` `version` with `actions` and pack libraries for
    /// `platforms`; `releases` lists every release for `latest` lookups.
    fn pack_release(
        fake: &Fake,
        releases: &mut Vec<serde_json::Value>,
        pack: &str,
        version: &str,
        actions: &[&str],
        platforms: &[(&str, &str)],
    ) {
        let base = format!("test://action_packs/{pack}/{version}");
        let mut names: Vec<String> = actions
            .iter()
            .map(|a| ActionIdentity::new(pack, version, a).unwrap().filename())
            .collect();
        names.extend(
            platforms
                .iter()
                .map(|(p, e)| pack_library_asset(pack, version, p, e)),
        );
        let checksums: String = names
            .iter()
            .map(|n| format!("{}  {n}\n", digest(n.as_bytes())))
            .collect();
        let mut assets: Vec<_> = names
            .iter()
            .map(|n| serde_json::json!({"name": n, "browser_download_url": format!("{base}/{n}")}))
            .collect();
        assets.push(serde_json::json!({"name": "checksums.txt", "browser_download_url": format!("{base}/checksums")}));
        let release = serde_json::json!({"tag_name": format!("action_packs/{pack}/v{version}"), "assets": assets});
        let mut data = fake.data.lock().unwrap();
        data.insert(format!("{base}/checksums"), checksums.into_bytes());
        data.insert(
            format!("https://api.github.com/repos/test/repo/releases/tags/action_packs%2F{pack}%2Fv{version}"),
            serde_json::to_vec(&release).unwrap(),
        );
        releases.push(release);
        data.insert(
            "https://api.github.com/repos/test/repo/releases?per_page=100&page=1".into(),
            serde_json::to_vec(&releases).unwrap(),
        );
    }

    fn app_client() -> ReleaseClient<Fake> {
        let fake = Fake::default();
        let mut releases = Vec::new();
        pack_release(
            &fake,
            &mut releases,
            "base",
            "0.3.3",
            &["identity"],
            &APP_PLATFORMS,
        );
        pack_release(
            &fake,
            &mut releases,
            "base",
            "0.3.4",
            &["identity"],
            &APP_PLATFORMS,
        );
        pack_release(
            &fake,
            &mut releases,
            "math_basics",
            "0.4.0",
            &["abs"],
            &APP_PLATFORMS,
        );
        pack_release(
            &fake,
            &mut releases,
            "math_basics",
            "0.4.1",
            &["abs", "add"],
            &APP_PLATFORMS,
        );
        pack_release(&fake, &mut releases, "old", "1.0.0", &["identity"], &[]);
        pack_release(
            &fake,
            &mut releases,
            "partial",
            "1.0.0",
            &["identity"],
            &APP_PLATFORMS[..7],
        );
        ReleaseClient {
            repository: "test/repo".into(),
            transport: fake,
        }
    }

    fn pipeline(body: &str) -> parser::ast::Pipeline {
        parser::parse(&format!("accept Tensor $t\n{body}")).unwrap()
    }

    #[test]
    fn app_packs_resolve_latest_and_let_a_pinned_version_serve_every_pipeline() {
        let client = app_client();
        let packs = app_packs(
            &[
                pipeline("$t >> base/latest/identity >> math_basics/latest/abs >> emit"),
                parser::parse(
                    "from base/0.3.3 import identity\naccept Tensor $t\n$t >> identity >> emit",
                )
                .unwrap(),
            ],
            &client,
        )
        .unwrap();
        assert_eq!(
            packs,
            BTreeMap::from([
                ("base".into(), "0.3.3".into()),
                ("math_basics".into(), "0.4.1".into())
            ])
        );
        // A whole-pack import needs the pack too.
        let packs = app_packs(
            &[parser::parse(
                "import math_basics/0.4.0\naccept Tensor $t\n$t >> math_basics.abs >> emit",
            )
            .unwrap()],
            &client,
        )
        .unwrap();
        assert_eq!(
            packs,
            BTreeMap::from([("math_basics".into(), "0.4.0".into())])
        );
    }

    #[test]
    fn app_pack_resolves_one_pack_for_an_app() {
        let client = app_client();
        let version =
            |pack, version| app_pack(&client, pack, version).map(|r| r.catalog.concrete_version);
        assert_eq!(version("math_basics", "latest").unwrap(), "0.4.1");
        assert_eq!(version("base", "0.3.3").unwrap(), "0.3.3");
        assert!(version("old", "latest")
            .unwrap_err()
            .contains("old 1.0.0 has no pack library"));
        assert!(version("partial", "1.0.0")
            .unwrap_err()
            .contains("no pack library for ios-arm64, ios-arm64-sim yet"));
    }

    #[test]
    fn app_packs_reject_what_an_app_cannot_bundle() {
        let client = app_client();
        let error = |source: &str| app_packs(&[pipeline(source)], &client).unwrap_err();
        assert!(
            error("$t >> old/latest/identity >> emit").contains("old 1.0.0 has no pack library")
        );
        assert!(error("$t >> partial/latest/identity >> emit")
            .contains("partial 1.0.0 has no pack library for ios-arm64, ios-arm64-sim yet"));
        let both = app_packs(
            &[
                pipeline("$t >> base/0.3.3/identity >> emit"),
                pipeline("$t >> base/0.3.4/identity >> emit"),
            ],
            &client,
        )
        .unwrap_err();
        assert!(both.contains("need base 0.3.3 and base 0.3.4"), "{both}");
        let plugin = app_packs(&[source("plugin opencv-bridge/0.1.0")], &client).unwrap_err();
        assert!(plugin.contains("apps cannot bundle plugins"), "{plugin}");
    }
}
