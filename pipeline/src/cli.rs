use std::env;
use std::fs::{self, File};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use crate::engine::collect_action_names;
use crate::resolver::ActionResolver;
use clap::{Parser, Subcommand};
use rich_rust::prelude::*;
use rich_rust::r#box::ROUNDED;
use rich_rust::renderables::markdown::Markdown;

#[derive(Parser, Debug)]
#[command(
    name = "morflow",
    version,
    about = "Morflow - High-performance modular dataflow pipeline engine for media and tensor computing"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Pre-downloads all actions required by a .morf pipeline ahead of time for offline execution
    Prep {
        /// Path to the .morf pipeline definition file
        file: PathBuf,

        /// Custom action cache directory (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Force re-download even if action is already cached locally
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
        /// Full action path '<package>/<version>/<action>' (e.g. image_essentials/latest/color_adjust or base/0.1.2/identity)
        action: String,
    },

    /// Performs fuzzy search for actions by name and returns the top 5 full action paths
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
        /// Full action path (e.g. image_essentials/latest/color_adjust, audio_essentials/gain)
        action: String,

        /// Custom action cache directory (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Force re-download even if action is already installed
        #[arg(short, long)]
        force: bool,
    },

    /// Checks a .morf pipeline for syntax correctness, SSA compliance, action dependencies, and data type flow
    Check {
        /// Path to the .morf pipeline definition file
        file: PathBuf,

        /// Custom action cache directory (defaults to MORFLOW_ACTIONS_PATH or ~/.morflow/actions)
        #[arg(short, long)]
        path: Option<PathBuf>,
    },
}

// Built-in catalog of Morflow action packs and actions
pub(crate) const KNOWN_ACTIONS: &[(&str, &str)] = &[
    ("base", "identity"),
    ("base", "to_tensor"),
    ("audio_essentials", "to_audio"),
    ("audio_essentials", "to_pcm"),
    ("audio_essentials", "to_wav"),
    ("audio_essentials", "gain"),
    ("audio_essentials", "normalize"),
    ("audio_essentials", "biquad_filter"),
    ("audio_essentials", "compressor"),
    ("audio_essentials", "limiter"),
    ("audio_essentials", "noise_gate"),
    ("audio_essentials", "stereo_widen"),
    ("audio_essentials", "resample"),
    ("audio_essentials", "stft"),
    ("audio_essentials", "delay"),
    ("image_essentials", "to_image"),
    ("image_essentials", "resize"),
    ("image_essentials", "crop"),
    ("image_essentials", "pad"),
    ("image_essentials", "color_adjust"),
    ("image_essentials", "gaussian_blur"),
    ("image_essentials", "edge_detect"),
    ("image_essentials", "sharpen"),
    ("image_essentials", "threshold"),
    ("image_essentials", "rotate"),
    ("image_essentials", "flip"),
    ("image_essentials", "blend"),
    ("image_essentials", "morphology"),
    ("tensor_essentials", "reshape"),
    ("tensor_essentials", "transpose"),
    ("tensor_essentials", "permute"),
    ("tensor_essentials", "squeeze"),
    ("tensor_essentials", "unsqueeze"),
    ("tensor_essentials", "flatten"),
    ("tensor_essentials", "concat"),
    ("tensor_essentials", "cast"),
    ("tensor_essentials", "repeat"),
    ("tensor_essentials", "roll"),
    ("math_essentials", "add"),
    ("math_essentials", "sub"),
    ("math_essentials", "mul"),
    ("math_essentials", "div"),
    ("math_essentials", "clamp"),
    ("math_essentials", "pow"),
    ("math_essentials", "abs"),
    ("math_essentials", "sign"),
    ("math_essentials", "sqrt"),
    ("math_essentials", "rsqrt"),
    ("math_essentials", "exp"),
    ("math_essentials", "log"),
    ("math_essentials", "sin"),
    ("math_essentials", "cos"),
    ("math_essentials", "tan"),
    ("math_essentials", "neg"),
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
    ("nn_essentials", "relu"),
    ("nn_essentials", "gelu"),
    ("nn_essentials", "silu"),
    ("nn_essentials", "sigmoid"),
    ("nn_essentials", "softmax"),
    ("nn_essentials", "log_softmax"),
    ("nn_essentials", "tanh"),
    ("nn_essentials", "leaky_relu"),
    ("nn_essentials", "layer_norm"),
    ("nn_essentials", "rms_norm"),
    ("nn_essentials", "cosine_similarity"),
    ("nn_essentials", "max_pool2d"),
    ("nn_essentials", "avg_pool2d"),
    ("linalg_essentials", "matmul"),
    ("linalg_essentials", "dot"),
    ("linalg_essentials", "outer"),
    ("linalg_essentials", "trace"),
    ("linalg_essentials", "diag"),
    ("linalg_essentials", "inv"),
    ("linalg_essentials", "det"),
    ("linalg_essentials", "qr"),
    ("linalg_essentials", "cholesky"),
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
        ("linux-x86_64", "so")
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

fn normalize_pack_name(pack: &str) -> String {
    let p = pack.trim().to_lowercase();
    match p.as_str() {
        "audio_essential" | "audio_essentials" => "audio_essentials".to_string(),
        "image_essential" | "image_essentials" => "image_essentials".to_string(),
        "tensor_essential" | "tensor_essentials" => "tensor_essentials".to_string(),
        "math_essential" | "math_essentials" => "math_essentials".to_string(),
        "tensor_stat" | "tensor_stats" => "tensor_stats".to_string(),
        "nn_essential" | "nn_essentials" => "nn_essentials".to_string(),
        "linalg_essential" | "linalg_essentials" => "linalg_essentials".to_string(),
        "base" => "base".to_string(),
        other if !other.is_empty() => other.to_string(),
        _ => "base".to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TargetPath {
    Action {
        pack: String,
        version: String,
        action: String,
    },
    Package {
        pack: String,
        version: String,
    },
}

fn parse_target_path(path_str: &str) -> Result<TargetPath, String> {
    let clean = path_str.replace("::", "/");
    let parts: Vec<&str> = if clean.contains('/') {
        clean
            .split('/')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    } else if clean.contains('.') {
        clean
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        vec![clean.trim()]
    };

    if parts.len() >= 3 {
        // e.g. image_essentials/latest/color_adjust or base/0.1.0/identity
        let pack = normalize_pack_name(parts[0]);
        let action = parts[parts.len() - 1].to_string();
        let raw_version = parts[1..parts.len() - 1].join("/");
        let version = raw_version.trim_start_matches('v').to_string();
        Ok(TargetPath::Action {
            pack,
            version,
            action,
        })
    } else if parts.len() == 2 {
        // e.g. base/0.1.0 or image_essentials/latest
        let pack = normalize_pack_name(parts[0]);
        let version = parts[1].trim_start_matches('v').to_string();
        Ok(TargetPath::Package { pack, version })
    } else {
        let hint_pack = parts.first().copied().unwrap_or("base");
        let hint_act = parts.last().copied().unwrap_or("identity");
        Err(format!(
            "Invalid target '{}'. Expected full action path '<package>/<version>/<action>' (e.g. '{}/latest/{}') or package path '<package>/<version>' (e.g. '{}/latest' or '{}/0.1.0').",
            path_str, hint_pack, hint_act, hint_pack, hint_pack
        ))
    }
}

fn get_actions_for_pack(pack: &str) -> Vec<&'static str> {
    let mut actions = Vec::new();
    for &(p, act) in KNOWN_ACTIONS {
        if p == pack {
            actions.push(act);
        }
    }
    actions
}

pub(crate) fn resolve_repo() -> String {
    env::var("MORFLOW_REPO").unwrap_or_else(|_| "JiraPit/Morflow".to_string())
}

fn fetch_latest_pack_version(pack: &str, repo: &str) -> Result<String, String> {
    let url = format!("https://api.github.com/repos/{}/releases", repo);
    let prefix_v = format!("action_packs/{}/v", pack);
    let prefix_no_v = format!("action_packs/{}/", pack);

    let response = ureq::get(&url)
        .set("User-Agent", "Morflow-CLI/0.1.2")
        .call()
        .map_err(|e| format!("failed to query releases for pack '{}': {}", pack, e))?;

    let body = response
        .into_string()
        .map_err(|e| format!("failed to read releases response: {}", e))?;

    let mut search_idx = 0;
    while let Some(pos) = body[search_idx..].find("\"tag_name\"") {
        let absolute_pos = search_idx + pos;
        let remainder = &body[absolute_pos..];
        if let Some(colon_pos) = remainder.find(':') {
            let after_colon = &remainder[colon_pos + 1..];
            if let Some(first_quote) = after_colon.find('"') {
                let val_slice = &after_colon[first_quote + 1..];
                if let Some(second_quote) = val_slice.find('"') {
                    let tag = &val_slice[..second_quote];
                    if tag.starts_with(&prefix_v) {
                        let ver = &tag[prefix_v.len()..];
                        if !ver.is_empty() {
                            return Ok(ver.to_string());
                        }
                    } else if tag.starts_with(&prefix_no_v) {
                        let ver = tag[prefix_no_v.len()..].trim_start_matches('v');
                        if !ver.is_empty() {
                            return Ok(ver.to_string());
                        }
                    }
                }
            }
        }
        search_idx = absolute_pos + 10;
    }

    Err(format!(
        "no released version found for pack '{}' in repo '{}'",
        pack, repo
    ))
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
        Err(err) => {
            let console = Console::new();
            console.print(&format!("[bold red]Error:[/] {}", err));
            1
        }
    }
}

fn run_command(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Check { file, path } => {
            crate::check::check_pipeline(&file, path.as_deref())?;
        }

        Commands::Prep { file, path, force } => {
            if !file.exists() {
                return Err(format!("Pipeline file '{}' does not exist.", file.display()).into());
            }

            let source = fs::read_to_string(&file)?;
            let pipeline = parser::parse(&source).map_err(|errs| {
                format!(
                    "Failed to parse pipeline {}: {}",
                    file.display(),
                    errs.iter()
                        .map(parser::format_error)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

            let resolver = ActionResolver::from_imports(&pipeline.imports);
            let action_names = collect_action_names(&pipeline.statements);

            let console = Console::new();

            if action_names.is_empty() {
                console.print(&format!(
                    "[yellow]⚠ No action calls found in '{}'. Nothing to prepare.[/]",
                    file.display()
                ));
                return Ok(());
            }

            let repo = resolve_repo();
            let (platform, ext) = get_host_platform();
            let cache_dir = resolve_action_cache_dir(path);
            fs::create_dir_all(&cache_dir)?;

            console.rule(Some("Morflow Action Pre-Downloader"));
            console.print(&format!(
                "  [bold cyan]Pipeline:[/]        [green]{}[/]",
                file.display()
            ));
            console.print(&format!(
                "  [bold cyan]Host Platform:[/]   [yellow]{}[/] (.{})",
                platform, ext
            ));
            console.print(&format!(
                "  [bold cyan]Cache Directory:[/] [dim]{}[/]",
                cache_dir.display()
            ));
            console.print(&format!(
                "  [bold cyan]Repository:[/]      [blue]{}[/]",
                repo
            ));
            console.print("");

            let mut prep_table = Table::new()
                .box_style(&ROUNDED)
                .border_style(Style::parse("bright_cyan").unwrap_or_default())
                .header_style(Style::parse("bold white on blue").unwrap_or_default())
                .with_column(Column::new("Action Pack").no_wrap())
                .with_column(Column::new("Action").no_wrap())
                .with_column(
                    Column::new("Version")
                        .width(10)
                        .justify(JustifyMethod::Center),
                )
                .with_column(
                    Column::new("Status")
                        .width(18)
                        .justify(JustifyMethod::Center),
                );

            let mut prepared_count = 0;
            let mut skipped_count = 0;

            for action_name in action_names {
                let (target_pack, real_action_name) = resolver.resolve(&action_name);
                let pack = target_pack.unwrap_or_else(|| "base".to_string());
                let action_version = fetch_latest_pack_version(&pack, &repo)?;

                let pack_dir = cache_dir.join(&pack);
                fs::create_dir_all(&pack_dir)?;

                let target_file_pack =
                    pack_dir.join(format!("{}_action.{}", real_action_name, ext));
                let target_file_root =
                    cache_dir.join(format!("{}_action.{}", real_action_name, ext));

                if !force && (target_file_pack.exists() || target_file_root.exists()) {
                    prep_table.add_row_markup([
                        pack.as_str(),
                        real_action_name.as_str(),
                        format!("v{}", action_version).as_str(),
                        "[bold green]✓ Cached[/]",
                    ]);
                    skipped_count += 1;
                    continue;
                }

                // Build candidate release download URLs
                let binary_filename = format!(
                    "{}_action-{}-{}.{}",
                    real_action_name, action_version, platform, ext
                );

                let urls = [
                    format!(
                        "https://github.com/{}/releases/download/action_packs%2F{}/v{}/{}",
                        repo, pack, action_version, binary_filename
                    ),
                    format!(
                        "https://github.com/{}/releases/download/action_packs/{}/v{}/{}",
                        repo, pack, action_version, binary_filename
                    ),
                ];

                let mut downloaded = false;
                for url in &urls {
                    match ureq::get(url).call() {
                        Ok(response) => {
                            let mut bytes = Vec::new();
                            response.into_reader().read_to_end(&mut bytes)?;

                            let mut file_pack = File::create(&target_file_pack)?;
                            file_pack.write_all(&bytes)?;

                            let mut file_root = File::create(&target_file_root)?;
                            file_root.write_all(&bytes)?;

                            prep_table.add_row_markup([
                                pack.as_str(),
                                real_action_name.as_str(),
                                format!("v{}", action_version).as_str(),
                                "[bold cyan]↓ Downloaded[/]",
                            ]);
                            downloaded = true;
                            prepared_count += 1;
                            break;
                        }
                        Err(_) => continue,
                    }
                }

                if !downloaded {
                    // Check local repository builds
                    let local_candidates = [
                        PathBuf::from(format!(
                            "target/release/actions/{}/{}_action.{}",
                            pack, real_action_name, ext
                        )),
                        PathBuf::from(format!(
                            "target/release/actions/{}_action.{}",
                            real_action_name, ext
                        )),
                        PathBuf::from(format!(
                            "actions/{}/{}/target/release/lib{}.{}",
                            pack, real_action_name, real_action_name, ext
                        )),
                    ];

                    let mut copied_local = false;
                    for cand in &local_candidates {
                        if cand.exists() {
                            fs::copy(cand, &target_file_pack)?;
                            fs::copy(cand, &target_file_root)?;
                            prep_table.add_row_markup([
                                pack.as_str(),
                                real_action_name.as_str(),
                                format!("v{}", action_version).as_str(),
                                "[bold magenta]⚡ Local Build[/]",
                            ]);
                            copied_local = true;
                            prepared_count += 1;
                            break;
                        }
                    }

                    if !copied_local {
                        prep_table.add_row_markup([
                            pack.as_str(),
                            real_action_name.as_str(),
                            format!("v{}", action_version).as_str(),
                            "[bold red]✗ Not Found[/]",
                        ]);
                    }
                }
            }

            console.print_renderable(&prep_table);
            console.print("");
            console.print(&format!(
                "[bold green]✓ Pipeline Ready:[/] {} action(s) prepared, {} already cached in [dim]{}[/].",
                prepared_count, skipped_count, cache_dir.display()
            ));
        }

        Commands::Clean { path } => {
            let console = Console::new();
            let cache_dir = resolve_action_cache_dir(path);
            console.rule(Some("Morflow Action Cache Cleaner"));
            console.print(&format!(
                "  [bold cyan]Target Directory:[/] [dim]{}[/]",
                cache_dir.display()
            ));
            console.print("");

            if !cache_dir.exists() {
                console.print("[yellow]⚠ Cache directory does not exist. Nothing to clean.[/]");
                return Ok(());
            }

            let mut deleted_files = 0;
            if let Ok(entries) = fs::read_dir(&cache_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        let _ = fs::remove_file(&p);
                        deleted_files += 1;
                    } else if p.is_dir() {
                        let _ = fs::remove_dir_all(&p);
                        deleted_files += 1;
                    }
                }
            }

            console.print(&format!(
                "[bold green]✓ Successfully cleaned[/] {} items from [dim]{}[/].",
                deleted_files,
                cache_dir.display()
            ));
        }

        Commands::Spec { action } => {
            let (pack, path_version, action_name) = match parse_target_path(&action) {
                Ok(TargetPath::Action {
                    pack,
                    version,
                    action,
                }) => (pack, version, action),
                Ok(TargetPath::Package { pack, .. }) => {
                    return Err(format!(
                        "`morflow spec` requires a 3-term action path '<package>/<version>/<action>' (e.g. '{}/latest/<action>'). A package path has no specification.",
                        pack
                    )
                    .into());
                }
                Err(err) => {
                    return Err(err.into());
                }
            };
            let repo = resolve_repo();
            let version = if path_version != "latest" && !path_version.is_empty() {
                path_version.to_string()
            } else {
                fetch_latest_pack_version(&pack, &repo)?
            };

            let render_spec = |content: &str| {
                let console = Console::new();
                if std::io::stdout().is_terminal() {
                    console.rule(Some(&format!(
                        "Action Specification: {}/{}/{}",
                        pack, version, action_name
                    )));
                    let md = Markdown::new(content);
                    console.print_renderable(&md);
                } else {
                    print!("{}", content);
                }
            };

            // 1. Check local cache directory for this pack & action
            let cache_spec = resolve_action_cache_dir(None)
                .join(&pack)
                .join(&action_name)
                .join("SPEC.md");
            if cache_spec.exists() {
                if let Ok(content) = fs::read_to_string(&cache_spec) {
                    render_spec(&content);
                    return Ok(());
                }
            }

            // 2. If in local repo/dev environment, check workspace file
            let local_candidates = [
                PathBuf::from(format!("actions/{}/{}/SPEC.md", pack, action_name)),
                PathBuf::from(format!("../actions/{}/{}/SPEC.md", pack, action_name)),
            ];

            for cand in &local_candidates {
                if cand.exists() {
                    if let Ok(content) = fs::read_to_string(cand) {
                        render_spec(&content);
                        return Ok(());
                    }
                }
            }

            // 3. Fetch version-specific SPEC.md from GitHub Release assets
            let release_urls = [
                format!(
                    "https://github.com/{}/releases/download/action_packs%2F{}/v{}/{}_SPEC.md",
                    repo, pack, version, action_name
                ),
                format!(
                    "https://github.com/{}/releases/download/action_packs/{}/v{}/{}_SPEC.md",
                    repo, pack, version, action_name
                ),
            ];

            for url in &release_urls {
                if let Ok(response) = ureq::get(url).set("User-Agent", "Morflow-CLI/0.1.2").call() {
                    let mut content = String::new();
                    if response.into_reader().read_to_string(&mut content).is_ok() {
                        render_spec(&content);
                        return Ok(());
                    }
                }
            }

            return Err(format!(
                "SPEC.md not found for action '{}/v{}/{}' (checked local paths and release assets).",
                pack, version, action_name
            )
            .into());
        }

        Commands::Search { query, limit } => {
            let q = query.to_lowercase();
            let mut matches: Vec<(&'static str, &'static str, usize, usize)> = KNOWN_ACTIONS
                .iter()
                .filter_map(|&(pack, act)| {
                    let act_lower = act.to_lowercase();
                    act_lower.find(&q).map(|pos| (pack, act, pos, act.len()))
                })
                .collect();

            // Rank by: 1) earlier substring position, 2) shorter action length, 3) alphabetical
            matches.sort_by(|a, b| {
                a.2.cmp(&b.2)
                    .then_with(|| a.3.cmp(&b.3))
                    .then_with(|| a.1.cmp(b.1))
                    .then_with(|| a.0.cmp(b.0))
            });

            let top_matches: Vec<_> = matches.into_iter().take(limit).collect();
            let console = Console::new();

            if top_matches.is_empty() {
                console.print(&format!(
                    "[yellow]⚠ No matching actions found for query[/] '[bold]{}[/]'.",
                    query
                ));
                console.print("[dim]Available packs: base, audio_essentials, image_essentials, tensor_essentials, math_essentials, tensor_stats, nn_essentials, linalg_essentials[/]");
            } else {
                console.rule(Some(&format!("Action Catalog Search: '{}'", query)));
                let mut search_table = Table::new()
                    .box_style(&ROUNDED)
                    .border_style(Style::parse("bright_cyan").unwrap_or_default())
                    .header_style(Style::parse("bold white on blue").unwrap_or_default())
                    .with_column(Column::new("Package").no_wrap())
                    .with_column(Column::new("Action").no_wrap())
                    .with_column(Column::new("Full Target Path").no_wrap());

                for (pack, act, _, _) in top_matches {
                    let full_path = format!("{}/latest/{}", pack, act);
                    search_table.add_row_markup([
                        pack,
                        act,
                        &format!("[bold green]{}[/]", full_path),
                    ]);
                }

                console.print_renderable(&search_table);
                console.print("");
                console.print("[dim]Tip: Run [bold cyan]morflow spec <target_path>[/] to view action documentation.[/]");
            }
        }

        Commands::List { path } => {
            let cache_dir = resolve_action_cache_dir(path);
            let mut found_actions = Vec::new();

            let (_platform, ext) = get_host_platform();
            let suffix = format!("_action.{}", ext);

            if cache_dir.exists() {
                if let Ok(entries) = fs::read_dir(&cache_dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.is_dir() {
                            let pack_name = p
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("")
                                .to_string();
                            if let Ok(sub_entries) = fs::read_dir(&p) {
                                for sub_entry in sub_entries.flatten() {
                                    let sub_p = sub_entry.path();
                                    if let Some(file_name) =
                                        sub_p.file_name().and_then(|n| n.to_str())
                                    {
                                        if file_name.ends_with(&suffix) {
                                            let act_name =
                                                &file_name[..file_name.len() - suffix.len()];
                                            let full_path =
                                                format!("{}/latest/{}", pack_name, act_name);
                                            if !found_actions.contains(&full_path) {
                                                found_actions.push(full_path);
                                            }
                                        }
                                    }
                                }
                            }
                        } else if p.is_file() {
                            if let Some(file_name) = p.file_name().and_then(|n| n.to_str()) {
                                if file_name.ends_with(&suffix) {
                                    let act_name = &file_name[..file_name.len() - suffix.len()];
                                    // Resolve pack
                                    let mut pack = "base".to_string();
                                    for &(k_pack, k_act) in KNOWN_ACTIONS {
                                        if k_act == act_name {
                                            pack = k_pack.to_string();
                                            break;
                                        }
                                    }
                                    let full_path = format!("{}/latest/{}", pack, act_name);
                                    if !found_actions.contains(&full_path) {
                                        found_actions.push(full_path);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Also check local workspace builds if available
            let dev_target = Path::new("target/release/actions");
            if dev_target.exists() {
                if let Ok(entries) = fs::read_dir(dev_target) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.is_dir() {
                            let pack_name = p
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("")
                                .to_string();
                            if let Ok(sub_entries) = fs::read_dir(&p) {
                                for sub_entry in sub_entries.flatten() {
                                    let sub_p = sub_entry.path();
                                    if let Some(file_name) =
                                        sub_p.file_name().and_then(|n| n.to_str())
                                    {
                                        if file_name.ends_with(&suffix) {
                                            let act_name =
                                                &file_name[..file_name.len() - suffix.len()];
                                            let full_path =
                                                format!("{}/latest/{}", pack_name, act_name);
                                            if !found_actions.contains(&full_path) {
                                                found_actions.push(full_path);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            found_actions.sort();
            let console = Console::new();

            if found_actions.is_empty() {
                console.print(&format!(
                    "[yellow]⚠ No installed actions found in cache[/] [dim]({})[/].",
                    cache_dir.display()
                ));
                console.print("[dim]Run [bold cyan]morflow prep <pipeline.morf>[/] to download required actions.[/]");
            } else {
                console.rule(Some("Installed Actions"));
                let width = console.width();
                let mut list_table = Table::new()
                    .box_style(&ROUNDED)
                    .border_style(Style::parse("bright_cyan").unwrap_or_default())
                    .header_style(Style::parse("bold white on blue").unwrap_or_default())
                    .with_column(Column::new("Package"))
                    .with_column(Column::new("Action"));

                let show_full_path = width >= 90;
                if show_full_path {
                    list_table = list_table
                        .with_column(Column::new("Target Path"))
                        .with_column(
                            Column::new("SPEC.md")
                                .width(10)
                                .justify(JustifyMethod::Center),
                        );
                } else {
                    list_table = list_table.with_column(
                        Column::new("SPEC.md")
                            .width(10)
                            .justify(JustifyMethod::Center),
                    );
                }

                for act_path in &found_actions {
                    let parts: Vec<&str> = act_path.split('/').collect();
                    let (pack, act) = if parts.len() >= 3 {
                        (parts[0], parts[2])
                    } else {
                        ("base", act_path.as_str())
                    };
                    let spec_file = cache_dir.join(pack).join(act).join("SPEC.md");
                    let spec_status = if spec_file.exists() {
                        "[bold green]✓[/]"
                    } else {
                        "[dim]-[/]"
                    };
                    if show_full_path {
                        list_table.add_row_markup([
                            pack,
                            act,
                            &format!("[green]{}[/]", act_path),
                            spec_status,
                        ]);
                    } else {
                        list_table.add_row_markup([pack, act, spec_status]);
                    }
                }

                console.print_renderable(&list_table);
                console.print("");
                console.print(&format!("[dim]Cache directory: {}[/]", cache_dir.display()));
            }
        }

        Commands::Install {
            action,
            path,
            force,
        } => {
            let target = match parse_target_path(&action) {
                Ok(res) => res,
                Err(err) => {
                    return Err(err.into());
                }
            };
            let repo = resolve_repo();
            let (platform, ext) = get_host_platform();
            let cache_dir = resolve_action_cache_dir(path);

            match target {
                TargetPath::Action {
                    pack,
                    version: path_version,
                    action: action_name,
                } => {
                    let version = if path_version != "latest" && !path_version.is_empty() {
                        path_version.to_string()
                    } else {
                        fetch_latest_pack_version(&pack, &repo)?
                    };

                    let pack_dir = cache_dir.join(&pack);
                    fs::create_dir_all(&pack_dir)?;

                    let target_file_pack = pack_dir.join(format!("{}_action.{}", action_name, ext));
                    let target_file_root =
                        cache_dir.join(format!("{}_action.{}", action_name, ext));

                    let console = Console::new();

                    if !force && (target_file_pack.exists() || target_file_root.exists()) {
                        console.print(&format!(
                            "[yellow]⚠ Action[/] [bold]{}/latest/{}[/] is already installed in [dim]{}[/]. Use [bold]--force[/] to reinstall.",
                            pack, action_name, target_file_pack.display()
                        ));
                        return Ok(());
                    }

                    console.rule(Some("Morflow Action Installer"));
                    console.print(&format!(
                        "  [bold cyan]Action:[/]          [green]{}/latest/{}[/]",
                        pack, action_name
                    ));
                    console.print(&format!(
                        "  [bold cyan]Version:[/]         [yellow]v{}[/]",
                        version
                    ));
                    console.print(&format!(
                        "  [bold cyan]Host Platform:[/]   [yellow]{}[/] (.{})",
                        platform, ext
                    ));
                    console.print(&format!(
                        "  [bold cyan]Cache Directory:[/] [dim]{}[/]",
                        cache_dir.display()
                    ));
                    console.print(&format!(
                        "  [bold cyan]Repository:[/]      [blue]{}[/]",
                        repo
                    ));
                    console.print("");

                    console.print(&format!(
                        "  [bold cyan]↓ Downloading[/] [{}] {} v{}...",
                        pack, action_name, version
                    ));
                    let binary_filename =
                        format!("{}_action-{}-{}.{}", action_name, version, platform, ext);

                    let urls = [
                        format!(
                            "https://github.com/{}/releases/download/action_packs%2F{}/v{}/{}",
                            repo, pack, version, binary_filename
                        ),
                        format!(
                            "https://github.com/{}/releases/download/action_packs/{}/v{}/{}",
                            repo, pack, version, binary_filename
                        ),
                    ];

                    let mut downloaded = false;
                    for url in &urls {
                        match ureq::get(url).call() {
                            Ok(response) => {
                                let mut bytes = Vec::new();
                                response.into_reader().read_to_end(&mut bytes)?;

                                let mut file_pack = File::create(&target_file_pack)?;
                                file_pack.write_all(&bytes)?;

                                let mut file_root = File::create(&target_file_root)?;
                                file_root.write_all(&bytes)?;

                                console.print(&format!(
                                    "    [bold green]✓ Successfully installed[/] to [dim]{}[/]",
                                    target_file_pack.display()
                                ));
                                downloaded = true;
                                break;
                            }
                            Err(_) => continue,
                        }
                    }

                    if !downloaded {
                        // Check local repository builds
                        let local_candidates = [
                            PathBuf::from(format!(
                                "target/release/actions/{}/{}_action.{}",
                                pack, action_name, ext
                            )),
                            PathBuf::from(format!(
                                "target/release/actions/{}_action.{}",
                                action_name, ext
                            )),
                            PathBuf::from(format!(
                                "actions/{}/{}/target/release/lib{}.{}",
                                pack, action_name, action_name, ext
                            )),
                        ];

                        let mut copied_local = false;
                        for cand in &local_candidates {
                            if cand.exists() {
                                fs::copy(cand, &target_file_pack)?;
                                fs::copy(cand, &target_file_root)?;
                                console.print(&format!(
                                    "    [bold magenta]⚡ Copied local build artifact[/] from [dim]{}[/]",
                                    cand.display()
                                ));
                                copied_local = true;
                                break;
                            }
                        }

                        if !copied_local {
                            return Err(format!(
                                "Could not download remote binary from GitHub Release or find local artifact for [{}] {}.",
                                pack, action_name
                            )
                            .into());
                        }
                    }

                    // Also try to cache SPEC.md if available
                    let spec_dir = cache_dir.join(&pack).join(&action_name);
                    let _ = fs::create_dir_all(&spec_dir);
                    let target_spec = spec_dir.join("SPEC.md");
                    let local_spec =
                        PathBuf::from(format!("actions/{}/{}/SPEC.md", pack, action_name));
                    if local_spec.exists() {
                        let _ = fs::copy(&local_spec, &target_spec);
                    }

                    console.print(&format!(
                        "\n[bold green]✓ Installation complete:[/] [bold]{}/latest/{}[/] is ready for runtime use.\n",
                        pack, action_name
                    ));
                }
                TargetPath::Package {
                    pack,
                    version: path_version,
                } => {
                    let version = if path_version != "latest" && !path_version.is_empty() {
                        path_version.to_string()
                    } else {
                        fetch_latest_pack_version(&pack, &repo)?
                    };

                    let actions = get_actions_for_pack(&pack);
                    if actions.is_empty() {
                        return Err(format!("Unknown action package '{}'.", pack).into());
                    }

                    let pack_dir = cache_dir.join(&pack);
                    fs::create_dir_all(&pack_dir)?;

                    let console = Console::new();
                    console.rule(Some("Morflow Action Pack Installer"));
                    console.print(&format!(
                        "  [bold cyan]Package:[/]         [green]{}[/] ({} actions)",
                        pack,
                        actions.len()
                    ));
                    console.print(&format!(
                        "  [bold cyan]Version:[/]         [yellow]v{}[/]",
                        version
                    ));
                    console.print(&format!(
                        "  [bold cyan]Host Platform:[/]   [yellow]{}[/] (.{})",
                        platform, ext
                    ));
                    console.print(&format!(
                        "  [bold cyan]Cache Directory:[/] [dim]{}[/]",
                        cache_dir.display()
                    ));
                    console.print(&format!(
                        "  [bold cyan]Repository:[/]      [blue]{}[/]",
                        repo
                    ));
                    console.print("");

                    let mut pack_table = Table::new()
                        .box_style(&ROUNDED)
                        .border_style(Style::parse("bright_cyan").unwrap_or_default())
                        .header_style(Style::parse("bold white on blue").unwrap_or_default())
                        .with_column(Column::new("Action").no_wrap())
                        .with_column(
                            Column::new("Version")
                                .width(10)
                                .justify(JustifyMethod::Center),
                        )
                        .with_column(
                            Column::new("Status")
                                .width(18)
                                .justify(JustifyMethod::Center),
                        );

                    let mut installed_count = 0;
                    let mut cached_count = 0;

                    for action_name in &actions {
                        let target_file_pack =
                            pack_dir.join(format!("{}_action.{}", action_name, ext));
                        let target_file_root =
                            cache_dir.join(format!("{}_action.{}", action_name, ext));

                        if !force && (target_file_pack.exists() || target_file_root.exists()) {
                            pack_table.add_row_markup([
                                action_name,
                                format!("v{}", version).as_str(),
                                "[bold green]✓ Cached[/]",
                            ]);
                            cached_count += 1;
                            continue;
                        }

                        let binary_filename =
                            format!("{}_action-{}-{}.{}", action_name, version, platform, ext);

                        let urls = [
                            format!(
                                "https://github.com/{}/releases/download/action_packs%2F{}/v{}/{}",
                                repo, pack, version, binary_filename
                            ),
                            format!(
                                "https://github.com/{}/releases/download/action_packs/{}/v{}/{}",
                                repo, pack, version, binary_filename
                            ),
                        ];

                        let mut downloaded = false;
                        for url in &urls {
                            match ureq::get(url).call() {
                                Ok(response) => {
                                    let mut bytes = Vec::new();
                                    if response.into_reader().read_to_end(&mut bytes).is_ok() {
                                        if let Ok(mut file_pack) = File::create(&target_file_pack) {
                                            let _ = file_pack.write_all(&bytes);
                                        }
                                        if let Ok(mut file_root) = File::create(&target_file_root) {
                                            let _ = file_root.write_all(&bytes);
                                        }
                                        pack_table.add_row_markup([
                                            action_name,
                                            format!("v{}", version).as_str(),
                                            "[bold cyan]↓ Installed[/]",
                                        ]);
                                        downloaded = true;
                                        installed_count += 1;
                                        break;
                                    }
                                }
                                Err(_) => continue,
                            }
                        }

                        if !downloaded {
                            let local_candidates = [
                                PathBuf::from(format!(
                                    "target/release/actions/{}/{}_action.{}",
                                    pack, action_name, ext
                                )),
                                PathBuf::from(format!(
                                    "target/release/actions/{}_action.{}",
                                    action_name, ext
                                )),
                                PathBuf::from(format!(
                                    "actions/{}/{}/target/release/lib{}.{}",
                                    pack, action_name, action_name, ext
                                )),
                            ];

                            let mut copied_local = false;
                            for cand in &local_candidates {
                                if cand.exists() {
                                    let _ = fs::copy(cand, &target_file_pack);
                                    let _ = fs::copy(cand, &target_file_root);
                                    pack_table.add_row_markup([
                                        action_name,
                                        format!("v{}", version).as_str(),
                                        "[bold magenta]⚡ Local Build[/]",
                                    ]);
                                    installed_count += 1;
                                    copied_local = true;
                                    break;
                                }
                            }

                            if !copied_local {
                                pack_table.add_row_markup([
                                    action_name,
                                    format!("v{}", version).as_str(),
                                    "[bold red]✗ Not Found[/]",
                                ]);
                            }
                        }
                    }

                    console.print_renderable(&pack_table);
                    console.print("");
                    console.print(&format!(
                        "[bold green]✓ Action package '{}/latest' ready:[/] {} action(s) installed/updated, {} already cached in [dim]{}[/].",
                        pack, installed_count, cached_count, cache_dir.display()
                    ));
                }
            }
        }
    }

    Ok(())
}
