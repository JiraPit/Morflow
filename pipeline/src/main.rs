use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use pipeline::engine::collect_action_names;
use pipeline::resolver::ActionResolver;

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

        /// GitHub repository to download action binaries from
        #[arg(short, long, default_value = "JiraPit/Morflow")]
        repo: String,

        /// Action Pack version tag to download
        #[arg(short, long, default_value = "0.1.0")]
        action_version: String,

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
}

fn get_host_platform() -> (&'static str, &'static str) {
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

fn resolve_action_cache_dir(custom: Option<PathBuf>) -> PathBuf {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prep {
            file,
            path,
            repo,
            action_version,
            force,
        } => {
            if !file.exists() {
                eprintln!("Error: Pipeline file '{}' does not exist.", file.display());
                std::process::exit(1);
            }

            let source = fs::read_to_string(&file)?;
            let pipeline = parser::parse(&source).map_err(|errs| {
                format!(
                    "Failed to parse pipeline {}: {}",
                    file.display(),
                    errs.into_iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

            let resolver = ActionResolver::from_imports(&pipeline.imports);
            let action_names = collect_action_names(&pipeline.statements);

            if action_names.is_empty() {
                println!("No action calls found in '{}'. Nothing to prepare.", file.display());
                return Ok(());
            }

            let (platform, ext) = get_host_platform();
            let cache_dir = resolve_action_cache_dir(path);
            fs::create_dir_all(&cache_dir)?;

            println!("==================================================");
            println!(" Morflow Action Pre-Downloader (morflow prep)");
            println!(" Pipeline: {}", file.display());
            println!(" Host Platform: {} (.{})", platform, ext);
            println!(" Cache Directory: {}", cache_dir.display());
            println!(" Repository: {}", repo);
            println!("==================================================");

            let mut prepared_count = 0;
            let mut skipped_count = 0;

            for action_name in action_names {
                let (target_pack, real_action_name) = resolver.resolve(&action_name);
                let pack = target_pack.unwrap_or_else(|| "base".to_string());

                let pack_dir = cache_dir.join(&pack);
                fs::create_dir_all(&pack_dir)?;

                let target_file_pack = pack_dir.join(format!("{}_action.{}", real_action_name, ext));
                let target_file_root = cache_dir.join(format!("{}_action.{}", real_action_name, ext));

                if !force && (target_file_pack.exists() || target_file_root.exists()) {
                    println!("  [✓ Cached] [{}] {}", pack, real_action_name);
                    skipped_count += 1;
                    continue;
                }

                println!("  [↓ Downloading] [{}] {} v{}...", pack, real_action_name, action_version);

                // Build candidate release download URLs
                let binary_filename = format!("{}_action-{}-{}.{}", real_action_name, action_version, platform, ext);
                
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

                            println!("    ✓ Successfully cached to {}", target_file_pack.display());
                            downloaded = true;
                            prepared_count += 1;
                            break;
                        }
                        Err(_) => continue,
                    }
                }

                if !downloaded {
                    // Check if a local copy exists in target/release/actions or system paths
                    let local_candidates = [
                        PathBuf::from(format!("target/release/actions/{}/{}_action.{}", pack, real_action_name, ext)),
                        PathBuf::from(format!("target/release/actions/{}_action.{}", real_action_name, ext)),
                        PathBuf::from(format!("actions/{}/{}/target/release/lib{}.{}", pack, real_action_name, real_action_name, ext)),
                    ];

                    let mut copied_local = false;
                    for cand in &local_candidates {
                        if cand.exists() {
                            fs::copy(cand, &target_file_pack)?;
                            fs::copy(cand, &target_file_root)?;
                            println!("    ✓ Copied local build artifact from {}", cand.display());
                            copied_local = true;
                            prepared_count += 1;
                            break;
                        }
                    }

                    if !copied_local {
                        eprintln!(
                            "    ✗ Warning: Could not download remote binary from GitHub Release (HTTP 404) or find local artifact for [{}] {}.",
                            pack, real_action_name
                        );
                    }
                }
            }

            println!("\nSummary: {} action(s) prepared, {} already cached.", prepared_count, skipped_count);
            println!("All actions ready in {} for offline runtime execution.\n", cache_dir.display());
        }

        Commands::Clean { path } => {
            let cache_dir = resolve_action_cache_dir(path);
            println!("==================================================");
            println!(" Morflow Action Cache Cleaner (morflow clean)");
            println!(" Target Directory: {}", cache_dir.display());
            println!("==================================================");

            if !cache_dir.exists() {
                println!("Cache directory does not exist. Nothing to clean.");
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

            println!("✓ Cleaned {} items from {}", deleted_files, cache_dir.display());
        }
    }

    Ok(())
}
