use crate::artifact::*;
use crate::release::{spec_path, PreparedRelease, PublishedPack, ReleaseClient};
use clap::{Parser, Subcommand};
use rich_rust::prelude::*;
use rich_rust::r#box::ROUNDED;
use rich_rust::renderables::table::Cell;
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
        /// Full action path '<package>/<version>/<action>' (e.g. image_basics/latest/color_adjust or base/0.3.3/identity)
        action: String,
    },

    /// Search the actions published in the repository's GitHub releases by name
    Search {
        /// Search query (e.g. color, blur, resample, gain)
        query: String,

        /// Maximum number of results to return (default: 5)
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },

    /// Lists every action pack published in the repository's GitHub releases, or every action in one pack
    Catalog {
        /// Pack to list the actions of (e.g. audio_basics); omit to list every pack
        pack: Option<String>,
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
    let args: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();
    let rich_errors = match args.get(1).and_then(|arg| arg.to_str()) {
        Some("catalog" | "check" | "list" | "search") => true,
        Some("plugins") => matches!(
            args.get(2).and_then(|arg| arg.to_str()),
            Some("list" | "search")
        ),
        _ => false,
    };
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            if rich_errors && e.use_stderr() {
                print_error(&e.to_string());
            } else {
                let _ = e.print();
            }
            return e.exit_code();
        }
    };
    crate::release::show_progress(progress_message(&cli.command));
    match run_command(cli.command) {
        Ok(()) => 0,
        Err(e) => {
            if rich_errors {
                print_error(&e.to_string());
            } else {
                eprintln!("Error: {e}");
            }
            1
        }
    }
}
/// Parse a spec target, which must name one action as pack/version/action.
fn spec_target(action: &str) -> Result<ActionIdentity, String> {
    match parse_target_path(action) {
        Ok(TargetPath::Action(id)) => Ok(id),
        Err(e) if action.split('/').count() == 3 => Err(e),
        _ => Err(format!(
            "Invalid action '{action}'; spec requires pack/version/action, e.g. image_basics/latest/resize. Run morflow search <name> to find an action's pack."
        )),
    }
}
/// The spinner text shown while a command waits on GitHub.
fn progress_message(command: &Commands) -> Option<&'static str> {
    match command {
        Commands::Catalog { .. } => Some("Loading catalog..."),
        Commands::Spec { .. } => Some("Loading spec..."),
        Commands::Search { .. } => Some("Searching..."),
        Commands::Install { .. }
        | Commands::Plugins {
            command: PluginCommands::Install { .. },
        } => Some("Installing..."),
        Commands::Prep { .. } => Some("Preparing..."),
        Commands::Check { .. } => Some("Checking..."),
        _ => None,
    }
}
/// Rich command failures use stderr and preserve diagnostic text literally.
fn print_error(message: &str) {
    let console = Console::builder().file(Box::new(std::io::stderr())).build();
    let width = console.width().saturating_sub(6).max(1);
    let wrapped = Text::new(message)
        .wrap(width)
        .into_iter()
        .map(|line| line.plain().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let panel = Panel::from_text(&wrapped)
        .title(Text::styled(
            "Error",
            Style::parse("bold red").unwrap_or_default(),
        ))
        .box_style(&ROUNDED)
        .border_style(Style::parse("red").unwrap_or_default());
    console.print_renderable(&panel);
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
/// Render literal user/cache text as cells rather than interpreting it as markup.
fn print_catalog(
    title: &str,
    context: &str,
    columns: &[&str],
    rows: Vec<Vec<String>>,
    empty: &str,
) {
    let console = Console::new();
    console.print_text(&Text::styled(
        title,
        Style::parse("bold cyan").unwrap_or_default(),
    ));
    console.print_text(&Text::styled(
        context,
        Style::parse("dim").unwrap_or_default(),
    ));
    console.print("");
    if rows.is_empty() {
        console.print(&format!("[dim]{empty}[/]"));
        return;
    }
    let mut table = Table::new()
        .box_style(&ROUNDED)
        .border_style(Style::parse("dim").unwrap_or_default())
        .header_style(Style::parse("bold cyan").unwrap_or_default());
    for column in columns {
        table = table.with_column(Column::new(*column));
    }
    let count = rows.len();
    for row in rows {
        table.add_row_cells(row.into_iter().map(Cell::new));
    }
    console.print_renderable(&table);
    console.print(&format!(
        "[dim]{count} {}[/]",
        if count == 1 { "result" } else { "results" }
    ));
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
            let mut rows = Vec::new();
            for receipt in installed_receipts(&root)? {
                let _guard = CacheGuard::acquire(&root, &receipt.identity.to_string())?;
                let (receipt, _) = read_verified(&root, &receipt.identity)?;
                rows.push(vec![
                    format!(
                        "{}/{}/{}",
                        receipt.identity.pack, receipt.identity.version, receipt.identity.action
                    ),
                    receipt.concrete_version,
                    receipt.identity.platform,
                    receipt.repository,
                ]);
            }
            rows.sort();
            print_catalog(
                "Installed actions",
                &format!("Cache: {}", root.display()),
                &[
                    "Action / requested version",
                    "Release",
                    "Platform",
                    "Repository",
                ],
                rows,
                "No verified actions installed. Run morflow prep <file> to prepare a pipeline.",
            );
        }
        Commands::Spec { action } => {
            let id = spec_target(&action)?;
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
            let packs = published_packs()?;
            let rows = search_actions(&packs, &query, limit)
                .into_iter()
                .map(|(pack, action)| {
                    vec![
                        action.into(),
                        pack.into(),
                        format!("{pack}/latest/{action}"),
                    ]
                })
                .collect();
            print_catalog(
                "Action search",
                &format!("Query: {query}"),
                &["Action", "Pack", "Import path"],
                rows,
                "No matching actions. Try a broader search.",
            );
        }
        Commands::Catalog { pack: None } => {
            let rows = published_packs()?
                .into_iter()
                .map(|pack| {
                    vec![
                        pack.name.clone(),
                        pack.version,
                        pack.actions.len().to_string(),
                        format!("{}/latest", pack.name),
                    ]
                })
                .collect();
            print_catalog(
                "Action packs",
                &format!(
                    "Repository: {} · Run morflow catalog <pack> to list a pack's actions",
                    resolve_repo()
                ),
                &["Pack", "Latest", "Actions", "Import path"],
                rows,
                "No action packs are published for this platform.",
            );
        }
        Commands::Catalog { pack: Some(pack) } => {
            let name = pack.to_lowercase();
            let pack = published_packs()?
                .into_iter()
                .find(|pack| pack.name == name)
                .ok_or_else(|| {
                    format!("Unknown pack '{name}'. Run morflow catalog to see every pack.")
                })?;
            let rows = pack
                .actions
                .iter()
                .map(|action| vec![action.clone(), format!("{name}/latest/{action}")])
                .collect();
            print_catalog(
                "Pack actions",
                &format!(
                    "Pack: {name} {} · Run morflow spec <import path> for an action's parameters",
                    pack.version
                ),
                &["Action", "Import path"],
                rows,
                "No actions in this pack.",
            );
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

/// The action packs published in the repository's releases for the host platform.
fn published_packs() -> Result<Vec<PublishedPack>, String> {
    ReleaseClient::new(resolve_repo())?.published_packs(get_host_platform().0)
}

/// Actions whose names contain `query`, earliest and shortest match first, as (pack, action).
fn search_actions<'a>(
    packs: &'a [PublishedPack],
    query: &str,
    limit: usize,
) -> Vec<(&'a str, &'a str)> {
    let mut matches: Vec<_> = packs
        .iter()
        .flat_map(|pack| pack.actions.iter().map(move |action| (pack, action)))
        .filter_map(|(pack, action)| {
            action
                .find(query)
                .map(|pos| (pack.name.as_str(), action.as_str(), pos, action.len()))
        })
        .collect();
    matches.sort_by(|a, b| {
        a.2.cmp(&b.2)
            .then(a.3.cmp(&b.3))
            .then(a.1.cmp(b.1))
            .then(a.0.cmp(b.0))
    });
    matches
        .into_iter()
        .take(limit)
        .map(|(pack, action, _, _)| (pack, action))
        .collect()
}

// Plugin search uses the shipped catalog and works offline.
const KNOWN_PLUGINS: &[&str] = &["openblas", "opencv-bridge"];

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
            let rows = search_plugins(&query, limit)
                .into_iter()
                .map(|name| vec![name.into(), format!("{name}/latest")])
                .collect();
            print_catalog(
                "Plugin search",
                &format!("Query: {query}"),
                &["Plugin", "Declaration"],
                rows,
                "No matching plugins. Try a broader search.",
            );
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
                        rows.push(vec![
                            format!("{}/{}", receipt.identity.name, receipt.identity.version),
                            receipt.concrete_version,
                            receipt.identity.platform,
                            receipt.repository,
                        ]);
                    }
                }
            }
            rows.sort();
            print_catalog(
                "Installed plugins",
                &format!("Cache: {}", root.display()),
                &[
                    "Plugin / requested version",
                    "Release",
                    "Platform",
                    "Repository",
                ],
                rows,
                "No verified plugins installed. Run morflow prep <file> to prepare a pipeline.",
            );
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

    #[test]
    fn action_search_ranks_earliest_and_shortest_matches_first() {
        let pack = |name: &str, actions: &[&str]| PublishedPack {
            name: name.into(),
            version: "0.1.0".into(),
            actions: actions.iter().map(|a| a.to_string()).collect(),
        };
        let packs = [
            pack("audio_basics", &["gain", "resample"]),
            pack("image_basics", &["sample_grid", "color_adjust"]),
            pack("math_basics", &["add"]),
        ];
        assert_eq!(
            search_actions(&packs, "sample", 5),
            [
                ("image_basics", "sample_grid"),
                ("audio_basics", "resample")
            ]
        );
        assert_eq!(
            search_actions(&packs, "a", 2),
            [("math_basics", "add"), ("audio_basics", "gain")]
        );
        assert!(search_actions(&packs, "blur", 5).is_empty());

        let cli = Cli::try_parse_from(["morflow", "catalog"]).unwrap();
        assert!(matches!(cli.command, Commands::Catalog { pack: None }));
        let cli = Cli::try_parse_from(["morflow", "catalog", "audio_basics"]).unwrap();
        assert!(matches!(cli.command, Commands::Catalog { pack: Some(_) }));
    }

    #[test]
    fn spec_requires_a_full_action_path() {
        assert_eq!(
            spec_target("image_basics/latest/resize").unwrap().action,
            "resize"
        );
        for bad in ["resize", "image_basics/latest", "a/b/c/d"] {
            let err = spec_target(bad).unwrap_err();
            assert!(err.contains("spec requires pack/version/action"), "{err}");
            assert!(!err.contains("use pack/version or"), "{err}");
        }
        let err = spec_target("image_basics/nope/resize").unwrap_err();
        assert!(err.contains("semantic version"), "{err}");
    }
}
