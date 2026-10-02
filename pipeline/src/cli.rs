use crate::artifact::*;
use crate::release::{spec_path, PreparedRelease, ReleaseClient};
use clap::{Parser, Subcommand};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
#[derive(Parser, Debug)]
#[command(
    name = "morflow",
    version,
    about = "Morflow - Data Processing & Tensor Transformation Engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Manage installed runtime plugins
    #[command(
        after_help = "Discover plugins with morflow plugins search <query>.\nInstall all pipeline dependencies with morflow prep <file>.\nPlugins are separate from actions and external system libraries."
    )]
    Plugins {
        #[command(subcommand)]
        command: PluginCommands,
    },

    /// Pre-downloads all actions and plugins required by a .morf pipeline ahead of time for offline execution
    #[command(
        long_about = "Prepare the actions and plugins declared in a .morf pipeline for offline use. Verify published checksums and check that declared plugin versions satisfy action requirements. Each latest declaration is resolved again; exact versions remain installed alongside it. External system libraries are loaded during execution, not preparation."
    )]
    Prep {
        /// Path to the .morf pipeline definition file
        file: PathBuf,

        /// Custom action cache directory (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Custom plugin cache directory (defaults to MORFLOW_PLUGINS_PATH or ~/.morflow/plugins)
        #[arg(long)]
        plugins_path: Option<PathBuf>,

        /// Force re-download of required actions and plugins
        #[arg(short, long)]
        force: bool,
    },

    /// Cleans and removes all cached action binaries from the action path
    Clean {
        /// Custom action cache directory to clean (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,
    },

    /// Views the raw SPEC.md documentation for a specified action
    Spec {
        /// Full action path '<package>/<version>/<action>' (e.g. image_basics/latest/color_adjust or base/0.1.2/identity)
        action: String,
    },

    /// Search the shipped action catalog by name
    Search {
        /// Search query (e.g. color, blur, resample, gain)
        query: String,

        /// Maximum number of results to return (default: 5)
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },

    /// Lists all action paths installed locally in the action cache
    List {
        /// Custom action cache directory to inspect (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,
    },

    /// Installs a specific action binary into the local action cache
    Install {
        /// Full action or pack path (e.g. image_basics/latest/color_adjust, audio_basics/latest)
        action: String,

        /// Custom action cache directory (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Force re-download even if action is already installed
        #[arg(short, long)]
        force: bool,
    },

    /// Check syntax, variables, prepared dependencies, types, and shapes
    #[command(
        long_about = "Check a pipeline using prepared local action and plugin artifacts without network requests. Verify dependency versions and binary checksums, then check types and shapes. Shape checking does not load external plugin libraries such as OpenCV. Run prep first to populate the caches."
    )]
    Check {
        /// Path to the .morf pipeline definition file
        file: PathBuf,

        /// Custom action cache directory (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,
        /// Plugin cache directory (defaults to MORFLOW_PLUGINS_PATH or ~/.morflow/plugins)
        #[arg(long)]
        plugins_path: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum PluginCommands {
    /// Search known runtime plugins by name
    #[command(
        long_about = "Search the shipped plugin catalog by name. Search works offline and prints installable paths with explicit latest versions.",
        after_help = "Example: morflow plugins search opencv --limit 5"
    )]
    Search {
        /// Name or part of a name (e.g. opencv or bridge)
        query: String,
        /// Maximum number of results
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },
    /// Install a runtime plugin with an explicit version or latest
    #[command(
        long_about = "Download a plugin for this platform and verify its published checksum. Supply name/version or name/latest; versions are never inferred. This installs the Morflow plugin, while external libraries such as OpenCV must be installed separately. Use prep to install all dependencies declared in a pipeline.",
        after_help = "Examples:\n  morflow plugins install opencv-bridge/0.1.0\n  morflow plugins install opencv-bridge/latest --path ./plugins_cache"
    )]
    Install {
        /// Plugin name and explicit version (e.g. opencv-bridge/0.1.0)
        plugin: String,
        /// Plugin cache directory (defaults to MORFLOW_PLUGINS_PATH or ~/.morflow/plugins)
        #[arg(short, long)]
        path: Option<PathBuf>,
        /// Download again even if cached bytes match the published checksum
        #[arg(short, long)]
        force: bool,
    },

    /// List verified installed plugin versions
    List {
        /// Plugin cache directory (defaults to MORFLOW_PLUGINS_PATH or ~/.morflow/plugins)
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Remove cached plugins and catalogs; loaded pipelines keep their handles
    Clean {
        /// Plugin cache directory (defaults to MORFLOW_PLUGINS_PATH or ~/.morflow/plugins)
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

// Built-in catalog of Morflow action packs and actions
pub(crate) const KNOWN_ACTIONS: &[(&str, &str)] = &[
    ("tensor_blas", "concat"),
    ("tensor_blas", "repeat"),
    ("tensor_blas", "roll"),
    ("linalg_blas", "matmul"),
    ("linalg_blas", "dot"),
    ("linalg_blas", "outer"),
    ("linalg_blas", "inv"),
    ("linalg_blas", "det"),
    ("linalg_blas", "qr"),
    ("linalg_blas", "cholesky"),
    ("base", "identity"),
    ("base", "to_tensor"),
    ("audio_basics", "to_audio"),
    ("audio_basics", "to_pcm"),
    ("audio_basics", "to_wav"),
    ("audio_basics", "gain"),
    ("audio_basics", "normalize"),
    ("audio_basics", "biquad_filter"),
    ("audio_basics", "compressor"),
    ("audio_basics", "limiter"),
    ("audio_basics", "noise_gate"),
    ("audio_basics", "stereo_widen"),
    ("audio_basics", "resample"),
    ("audio_basics", "stft"),
    ("audio_basics", "delay"),
    ("image_opencv", "resize"),
    ("image_opencv", "rotate"),
    ("image_opencv", "gaussian_blur"),
    ("image_opencv", "morphology"),
    ("image_opencv", "edge_detect"),
    ("image_opencv", "sharpen"),
    ("image_basics", "to_image"),
    ("image_basics", "resize"),
    ("image_basics", "crop"),
    ("image_basics", "pad"),
    ("image_basics", "color_adjust"),
    ("image_basics", "gaussian_blur"),
    ("image_basics", "edge_detect"),
    ("image_basics", "sharpen"),
    ("image_basics", "threshold"),
    ("image_basics", "rotate"),
    ("image_basics", "flip"),
    ("image_basics", "blend"),
    ("image_basics", "morphology"),
    ("tensor_basics", "reshape"),
    ("tensor_basics", "transpose"),
    ("tensor_basics", "permute"),
    ("tensor_basics", "squeeze"),
    ("tensor_basics", "unsqueeze"),
    ("tensor_basics", "flatten"),
    ("tensor_basics", "concat"),
    ("tensor_basics", "cast"),
    ("tensor_basics", "repeat"),
    ("tensor_basics", "roll"),
    ("math_basics", "add"),
    ("math_basics", "sub"),
    ("math_basics", "mul"),
    ("math_basics", "div"),
    ("math_basics", "clamp"),
    ("math_basics", "pow"),
    ("math_basics", "abs"),
    ("math_basics", "sign"),
    ("math_basics", "sqrt"),
    ("math_basics", "rsqrt"),
    ("math_basics", "exp"),
    ("math_basics", "log"),
    ("math_basics", "sin"),
    ("math_basics", "cos"),
    ("math_basics", "tan"),
    ("math_basics", "neg"),
    ("tensor_stats", "sum"),
    ("tensor_stats", "mean"),
    ("tensor_stats", "var"),
    ("tensor_stats", "std"),
    ("tensor_stats", "min"),
    ("tensor_stats", "max"),
    ("tensor_stats", "argmin"),
    ("tensor_stats", "argmax"),
    ("tensor_stats", "norm"),
    ("tensor_stats", "cumsum"),
    ("nn_basics", "relu"),
    ("nn_basics", "gelu"),
    ("nn_basics", "silu"),
    ("nn_basics", "sigmoid"),
    ("nn_basics", "softmax"),
    ("nn_basics", "log_softmax"),
    ("nn_basics", "tanh"),
    ("nn_basics", "leaky_relu"),
    ("nn_basics", "layer_norm"),
    ("nn_basics", "rms_norm"),
    ("nn_basics", "cosine_similarity"),
    ("nn_basics", "max_pool2d"),
    ("nn_basics", "avg_pool2d"),
    ("linalg_basics", "matmul"),
    ("linalg_basics", "dot"),
    ("linalg_basics", "outer"),
    ("linalg_basics", "trace"),
    ("linalg_basics", "diag"),
    ("linalg_basics", "inv"),
    ("linalg_basics", "det"),
    ("linalg_basics", "qr"),
    ("linalg_basics", "cholesky"),
];

pub(crate) fn get_host_platform() -> (&'static str, &'static str) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        ("linux-x86_64", "so")
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        ("linux-aarch64", "so")
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        ("darwin-arm64", "dylib")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        ("darwin-x86_64", "dylib")
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        ("windows-x86_64", "dll")
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        ("unsupported", "bin")
    }
}

pub(crate) fn resolve_action_cache_dir(custom: Option<PathBuf>) -> PathBuf {
    if let Some(p) = custom {
        return p;
    }
    if let Ok(env_path) = env::var("MORFLOW_ACTIONS_PATH") {
        if !env_path.trim().is_empty() {
            return PathBuf::from(env_path);
        }
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".morflow").join("actions");
    }
    PathBuf::from(".morflow").join("actions")
}

#[derive(Debug)]
enum TargetPath {
    Action(ActionIdentity),
    Package { pack: String, version: String },
}
fn parse_target_path(input: &str) -> Result<TargetPath, String> {
    let input = input.replace("::", "/");
    let parts: Vec<&str> = input.split('/').collect();
    if parts.len() == 3 {
        return Ok(TargetPath::Action(ActionIdentity::new(
            parts[0], parts[1], parts[2],
        )?));
    }
    if parts.len() == 2 {
        component(parts[0])?;
        return Ok(TargetPath::Package {
            pack: parts[0].into(),
            version: normalize_version(parts[1])?,
        });
    }
    if parts.len() == 1 {
        if let Some((pack, rest)) = input.split_once('.') {
            component(pack)?;
            if let Ok(version) = normalize_version(rest) {
                return Ok(TargetPath::Package {
                    pack: pack.into(),
                    version,
                });
            }
            if let Some((version, action)) = rest.rsplit_once('.') {
                return Ok(TargetPath::Action(ActionIdentity::new(
                    pack, version, action,
                )?));
            }
        }
    }
    Err(format!(
        "Invalid target '{input}'; use pack/version or pack/version/action"
    ))
}
pub(crate) fn resolve_repo() -> String {
    env::var("MORFLOW_REPO").unwrap_or_else(|_| "JiraPit/Morflow".into())
}
pub fn run_cli<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    match run_command(cli.command) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}
fn report_install(
    client: &ReleaseClient,
    root: &Path,
    id: &ActionIdentity,
    release: &PreparedRelease,
    force: bool,
) -> Result<(), String> {
    let changed = client.install(root, id, release, force)?;
    println!(
        "{id} -> {}: {}",
        release.catalog.concrete_version,
        if changed {
            "downloaded and verified"
        } else {
            "checksum verified"
        }
    );
    Ok(())
}
fn run_command(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Plugins { command } => run_plugin_command(command)?,
        Commands::Check {
            file,
            path,
            plugins_path,
        } => crate::check::check_pipeline_with_plugins(
            &file,
            path.as_deref(),
            plugins_path.as_deref(),
        )?,
        Commands::Prep {
            file,
            path,
            plugins_path,
            force,
        } => {
            let source = fs::read_to_string(&file)?;
            let ast = parser::parse(&source).map_err(|errors| {
                errors
                    .iter()
                    .map(parser::format_error)
                    .collect::<Vec<_>>()
                    .join("\n")
            })?;
            let client = ReleaseClient::new(resolve_repo())?;
            let root = resolve_action_cache_dir(path);
            let plugin_root = crate::plugins::cache_dir(plugins_path);
            let result = crate::preparation::prepare(&ast, &client, &root, &plugin_root, force)?;
            println!(
                "Pipeline ready: {} verified action(s), {} verified plugin(s)",
                result.actions, result.plugins
            );
            println!("Actions: {}", root.display());
            if result.plugins > 0 {
                println!("Plugins: {}", plugin_root.display());
            }
        }
        Commands::Install {
            action,
            path,
            force,
        } => {
            let client = ReleaseClient::new(resolve_repo())?;
            let root = resolve_action_cache_dir(path);
            match parse_target_path(&action)? {
                TargetPath::Action(id) => {
                    let release = client.resolve(&id.pack, &id.version)?;
                    report_install(&client, &root, &id, &release, force)?;
                }
                TargetPath::Package { pack, version } => {
                    let release = client.resolve(&pack, &version)?;
                    let actions = release.catalog.actions(get_host_platform().0);
                    if actions.is_empty() {
                        return Err(format!(
                            "No action assets for {pack}/{version} on {}",
                            get_host_platform().0
                        )
                        .into());
                    }
                    for action in actions {
                        report_install(
                            &client,
                            &root,
                            &ActionIdentity::new(&pack, &version, &action)?,
                            &release,
                            force,
                        )?;
                    }
                }
            }
        }
        Commands::List { path } => {
            let root = resolve_action_cache_dir(path);
            let _maintenance = CacheGuard::shared(&root, "maintenance")?;
            for receipt in installed_receipts(&root)? {
                let _guard = CacheGuard::acquire(&root, &receipt.identity.to_string())?;
                let (receipt, _) = read_verified(&root, &receipt.identity)?;
                println!(
                    "{} -> {} [{}]",
                    receipt.identity, receipt.concrete_version, receipt.repository
                );
            }
        }
        Commands::Spec { action } => {
            let id = match parse_target_path(&action)? {
                TargetPath::Action(id) => id,
                _ => return Err("spec requires pack/version/action".into()),
            };
            let root = resolve_action_cache_dir(None);
            let path = id.path(&root);
            let (cached, installed_version) = {
                let _maintenance = CacheGuard::shared(&root, "maintenance")?;
                let _guard = CacheGuard::acquire(&root, &id.to_string())?;
                match read_verified(&root, &id) {
                    Ok((receipt, _)) => {
                        let cached =
                            ReleaseCatalog::read(&root, &id.pack, &receipt.concrete_version)
                                .ok()
                                .filter(|catalog| catalog.repository == receipt.repository)
                                .and_then(|catalog| {
                                    fs::read(spec_path(&path)).ok().filter(|bytes| {
                                        catalog
                                            .checksums
                                            .get(&format!("{}_SPEC.md", id.action))
                                            .is_some_and(|hash| *hash == digest(bytes))
                                    })
                                });
                        (cached, Some(receipt.concrete_version))
                    }
                    Err(_) => (None, None),
                }
            };
            let bytes = match cached {
                Some(bytes) => bytes,
                None => {
                    let client = ReleaseClient::new(resolve_repo())?;
                    let release = client.resolve(
                        &id.pack,
                        installed_version.as_deref().unwrap_or(&id.version),
                    )?;
                    client.spec(&release, &id.action)?
                }
            };
            print!("{}", String::from_utf8(bytes)?);
        }
        Commands::Search { query, limit } => {
            let query = query.to_lowercase();
            let mut matches: Vec<_> = KNOWN_ACTIONS
                .iter()
                .filter_map(|&(pack, action)| {
                    action
                        .find(&query)
                        .map(|pos| (pack, action, pos, action.len()))
                })
                .collect();
            matches.sort_by(|a, b| {
                a.2.cmp(&b.2)
                    .then(a.3.cmp(&b.3))
                    .then(a.1.cmp(b.1))
                    .then(a.0.cmp(b.0))
            });
            for (pack, action, _, _) in matches.into_iter().take(limit) {
                println!("{pack}/latest/{action}");
            }
        }
        Commands::Clean { path } => {
            let root = resolve_action_cache_dir(path);
            if !root.exists() {
                println!("Cache is empty");
                return Ok(());
            }
            // Exclude preparation/loading while maintenance removes artifacts.
            let _maintenance = CacheGuard::acquire(&root, "maintenance")?;
            for receipt in installed_receipts(&root)? {
                for path in [
                    receipt.identity.path(&root),
                    receipt_path(&receipt.identity.path(&root)),
                    spec_path(&receipt.identity.path(&root)),
                ] {
                    if path.exists() {
                        fs::remove_file(path)?;
                    }
                }
            }
            for pack in fs::read_dir(&root)? {
                let pack = pack?.path();
                let catalogs = pack.join(".catalogs");
                if catalogs.is_dir() {
                    for entry in fs::read_dir(catalogs)? {
                        let path = entry?.path();
                        if path.extension().is_some_and(|ext| ext == "json") {
                            fs::remove_file(path)?;
                        }
                    }
                }
            }
            let objects = root.join(".objects");
            if objects.is_dir() {
                for bucket in fs::read_dir(objects)? {
                    let bucket = bucket?.path();
                    if !bucket.is_dir() {
                        continue;
                    }
                    for entry in fs::read_dir(&bucket)? {
                        let path = entry?.path();
                        if path.is_file() {
                            match fs::remove_file(path) {
                                Ok(()) => {}
                                // Windows keeps mapped DLLs open. Existing pipelines
                                // retain those snapshots until their handles close.
                                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {}
                                Err(e) => return Err(e.into()),
                            }
                        }
                    }
                    if fs::read_dir(&bucket)?.next().is_none() {
                        fs::remove_dir(bucket)?;
                    }
                }
            }
            // Lock inodes stay in place for concurrent processes.
            println!(
                "Removed versioned artifacts and catalogs from {}",
                root.display()
            );
        }
    }
    Ok(())
}
fn installed_receipts(root: &Path) -> Result<Vec<ArtifactReceipt>, String> {
    let mut receipts = Vec::new();
    if !root.exists() {
        return Ok(receipts);
    }
    for pack in fs::read_dir(root).map_err(|e| e.to_string())? {
        let pack = pack.map_err(|e| e.to_string())?.path();
        if !pack.is_dir()
            || pack
                .file_name()
                .is_some_and(|s| s.to_string_lossy().starts_with('.'))
        {
            continue;
        }
        for entry in fs::read_dir(&pack).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "json") {
                let receipt: ArtifactReceipt = read_json(&path)?;
                ActionIdentity::for_platform(
                    &receipt.identity.pack,
                    &receipt.identity.version,
                    &receipt.identity.action,
                    &receipt.identity.platform,
                )?;
                let expected = receipt_path(&receipt.identity.path(root));
                if path != expected {
                    return Err(format!("Receipt path mismatch at {}", path.display()));
                }
                receipts.push(receipt);
            }
        }
    }
    receipts.sort_by_key(|r| r.identity.to_string());
    Ok(receipts)
}

// Like action search, plugin search uses the shipped catalog and works offline.
const KNOWN_PLUGINS: &[&str] = &["opencv-bridge"];

fn search_plugins(query: &str, limit: usize) -> Vec<&'static str> {
    let query = query.to_lowercase();
    let mut matches: Vec<_> = KNOWN_PLUGINS
        .iter()
        .filter_map(|&name| name.find(&query).map(|position| (name, position)))
        .collect();
    matches.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(a.0.len().cmp(&b.0.len()))
            .then(a.0.cmp(b.0))
    });
    matches
        .into_iter()
        .take(limit)
        .map(|(name, _)| name)
        .collect()
}

fn run_plugin_command(command: PluginCommands) -> Result<(), Box<dyn std::error::Error>> {
    use crate::plugins::{PluginIdentity, PluginReceipt};
    match command {
        PluginCommands::Search { query, limit } => {
            for name in search_plugins(&query, limit) {
                println!("{name}/latest");
            }
        }
        PluginCommands::Install {
            plugin,
            path,
            force,
        } => {
            let (name, version) = plugin
                .split_once('/')
                .ok_or("Plugin version is required; use name/version or name/latest")?;
            let id = PluginIdentity::new(name, version)?;
            let client = ReleaseClient::new(resolve_repo())?;
            let release = client.resolve_plugin(name, &id.version)?;
            let changed =
                client.install_plugin(&crate::plugins::cache_dir(path), &id, &release, force)?;
            println!(
                "{id} -> {}: {}",
                release.catalog.concrete_version,
                if changed {
                    "downloaded and verified"
                } else {
                    "checksum verified"
                }
            );
        }
        PluginCommands::List { path } => {
            let root = crate::plugins::cache_dir(path);
            let _maintenance = CacheGuard::shared(&root, "maintenance")?;
            let mut rows = Vec::new();
            if root.exists() {
                for directory in fs::read_dir(&root)? {
                    let directory = directory?.path();
                    if !directory.is_dir()
                        || directory
                            .file_name()
                            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                    {
                        continue;
                    }
                    for file in fs::read_dir(directory)? {
                        let path = file?.path();
                        if !path.file_name().is_some_and(|n| {
                            n.to_string_lossy().contains("_plugin-")
                                && n.to_string_lossy().ends_with(".json")
                        }) {
                            continue;
                        }
                        let receipt: PluginReceipt = read_json(&path)?;
                        if receipt_path(&receipt.identity.path(&root)) != path {
                            return Err("Plugin receipt is stored under the wrong identity".into());
                        }
                        let _guard = CacheGuard::acquire(&root, &receipt.identity.to_string())?;
                        let (receipt, _) = crate::plugins::read_verified(&root, &receipt.identity)?;
                        rows.push(format!(
                            "{} -> {} [{}]",
                            receipt.identity, receipt.concrete_version, receipt.repository
                        ));
                    }
                }
            }
            rows.sort();
            if rows.is_empty() {
                println!("No plugins installed in {}", root.display());
            } else {
                for row in rows {
                    println!("{row}");
                }
            }
        }
        PluginCommands::Clean { path } => {
            let root = crate::plugins::cache_dir(path);
            let _guard = CacheGuard::acquire(&root, "maintenance")?;
            if root.exists() {
                for entry in fs::read_dir(&root)? {
                    let path = entry?.path();
                    if path.is_dir()
                        && path
                            .file_name()
                            .is_some_and(|n| !n.to_string_lossy().starts_with('.'))
                    {
                        fs::remove_dir_all(path)?;
                    }
                }
            }
            println!("Removed cached plugins from {}", root.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plugin_search_matches_case_insensitively_and_obeys_limit() {
        assert_eq!(search_plugins("OpenCV", 5), vec!["opencv-bridge"]);
        assert!(search_plugins("unavailable", 5).is_empty());
        assert!(search_plugins("opencv", 0).is_empty());
    }

    #[test]
    fn parses_grouped_plugin_commands() {
        let cli =
            Cli::try_parse_from(["morflow", "plugins", "search", "opencv", "-l", "1"]).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Plugins {
                command: PluginCommands::Search { limit: 1, .. }
            }
        ));
        let cli = Cli::try_parse_from([
            "morflow",
            "plugins",
            "install",
            "opencv-bridge/latest",
            "-p",
            "cache",
            "-f",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Commands::Plugins {
                command: PluginCommands::Install {
                    force: true,
                    path: Some(_),
                    ..
                }
            }
        ));
        assert!(Cli::try_parse_from(["morflow", "plugins", "install"]).is_err());
        assert!(Cli::try_parse_from(["morflow", "search-plugin", "opencv"]).is_err());
        assert!(
            Cli::try_parse_from(["morflow", "install-plugin", "opencv-bridge/latest"]).is_err()
        );
    }
    #[test]
    fn parses_exact_versions_without_splitting_semver_dots() {
        for target in [
            "base/0.2.0/identity",
            "base.0.2.0.identity",
            "base::0.2.0::identity",
        ] {
            match parse_target_path(target).unwrap() {
                TargetPath::Action(id) => {
                    assert_eq!(id.pack, "base");
                    assert_eq!(id.version, "0.2.0");
                    assert_eq!(id.action, "identity");
                }
                _ => panic!("Expected action"),
            }
        }
        assert!(parse_target_path("base/identity").is_err());
        assert!(parse_target_path("base/../identity").is_err());
    }
}
