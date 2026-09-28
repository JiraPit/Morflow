#!/usr/bin/env node
/**
 * Morflow Node.js CLI
 * Provides 'prep' and 'clean' subcommands when installed via npm or run via npx.
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import process from 'node:process';

function getHostPlatform() {
  const platform = os.platform();
  const arch = os.arch();

  if (platform === 'linux') {
    return arch === 'arm64' ? ['linux-aarch64', 'so'] : ['linux-x86_64', 'so'];
  } else if (platform === 'darwin') {
    return arch === 'arm64' ? ['darwin-arm64', 'dylib'] : ['darwin-x86_64', 'dylib'];
  } else if (platform === 'win32') {
    return ['windows-x86_64', 'dll'];
  }
  return ['linux-x86_64', 'so'];
}

function resolveActionCacheDir(customPath) {
  if (customPath) {
    return path.resolve(customPath);
  }
  const envPath = process.env.MORFLOW_ACTIONS_PATH;
  if (envPath && envPath.trim()) {
    return path.resolve(envPath.trim());
  }
  return path.join(os.homedir(), '.morflow', 'actions');
}

function extractActionsFromMorf(source) {
  const imports = {};
  const actions = [];

  for (const rawLine of source.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const importMatch = line.match(/^(?:import|use)\s+([a-zA-Z0-9_]+)(?:\s+as\s+([a-zA-Z0-9_]+))?/);
    if (importMatch) {
      const pack = importMatch[1];
      const alias = importMatch[2] || pack;
      imports[alias] = pack;
    }

    const actionRegex = />>\s*([a-zA-Z0-9_.:]+)/g;
    let match;
    while ((match = actionRegex.exec(line)) !== null) {
      const actFull = match[1].split('(')[0].trim();
      if (actFull !== 'emit' && actFull !== 'resurface' && !actions.includes(actFull)) {
        actions.push(actFull);
      }
    }
  }

  const audioActions = new Set([
    'gain', 'normalize', 'biquad_filter', 'compressor', 'limiter',
    'noise_gate', 'stereo_widen', 'resample', 'stft', 'delay',
    'to_audio', 'to_pcm', 'to_wav'
  ]);

  const imageActions = new Set([
    'blend', 'color_adjust', 'crop', 'edge_detect', 'flip',
    'gaussian_blur', 'morphology', 'pad', 'resize', 'rotate',
    'sharpen', 'threshold', 'to_image'
  ]);

  return actions.map(act => {
    if (act.includes('::')) {
      const [pack, name] = act.split('::');
      return [pack, name];
    } else if (act.includes('.')) {
      const [pack, name] = act.split('.');
      return [pack, name];
    } else if (audioActions.has(act)) {
      return ['audio_essentials', act];
    } else if (imageActions.has(act)) {
      return ['image_essentials', act];
    } else {
      return ['base', act];
    }
  });
}

async function cmdPrep(args) {
  const filePath = path.resolve(args.file);
  if (!fs.existsSync(filePath)) {
    console.error(`Error: Pipeline file '${filePath}' does not exist.`);
    process.exit(1);
  }

  const source = fs.readFileSync(filePath, 'utf-8');
  const actions = extractActionsFromMorf(source);

  if (actions.length === 0) {
    console.log(`No action calls found in '${filePath}'. Nothing to prepare.`);
    return;
  }

  const [platformName, ext] = getHostPlatform();
  const cacheDir = resolveActionCacheDir(args.path);
  fs.mkdirSync(cacheDir, { recursive: true });

  console.log('==================================================');
  console.log(' Morflow Action Pre-Downloader (Node.js CLI)');
  console.log(` Pipeline: ${filePath}`);
  console.log(` Host Platform: ${platformName} (.${ext})`);
  console.log(` Cache Directory: ${cacheDir}`);
  console.log(` Repository: ${args.repo}`);
  console.log('==================================================');

  let prepared = 0;
  let cached = 0;

  for (const [pack, actionName] of actions) {
    const packDir = path.join(cacheDir, pack);
    fs.mkdirSync(packDir, { recursive: true });

    const targetFilePack = path.join(packDir, `${actionName}_action.${ext}`);
    const targetFileRoot = path.join(cacheDir, `${actionName}_action.${ext}`);

    if (!args.force && (fs.existsSync(targetFilePack) || fs.existsSync(targetFileRoot))) {
      console.log(`  [✓ Cached] [${pack}] ${actionName}`);
      cached++;
      continue;
    }

    console.log(`  [↓ Downloading] [${pack}] ${actionName} v${args.actionVersion}...`);
    const binaryFilename = `${actionName}_action-${args.actionVersion}-${platformName}.${ext}`;

    const urls = [
      `https://github.com/${args.repo}/releases/download/action_packs%2F${pack}%2Fv${args.actionVersion}/${binaryFilename}`,
      `https://github.com/${args.repo}/releases/download/action_packs/${pack}/v${args.actionVersion}/${binaryFilename}`,
    ];

    let downloaded = false;
    for (const url of urls) {
      try {
        const resp = await fetch(url, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.0' } });
        if (resp.ok) {
          const buffer = Buffer.from(await resp.arrayBuffer());
          fs.writeFileSync(targetFilePack, buffer);
          fs.writeFileSync(targetFileRoot, buffer);
          console.log(`    ✓ Successfully cached to ${targetFilePack}`);
          downloaded = true;
          prepared++;
          break;
        }
      } catch {}
    }

    if (!downloaded) {
      const localCandidates = [
        path.resolve(`target/release/actions/${pack}/${actionName}_action.${ext}`),
        path.resolve(`target/release/actions/${actionName}_action.${ext}`),
      ];

      let copied = false;
      for (const cand of localCandidates) {
        if (fs.existsSync(cand)) {
          fs.copyFileSync(cand, targetFilePack);
          fs.copyFileSync(cand, targetFileRoot);
          console.log(`    ✓ Copied local build artifact from ${cand}`);
          copied = true;
          prepared++;
          break;
        }
      }

      if (!copied) {
        console.error(`    ✗ Warning: Could not download remote binary or find local artifact for [${pack}] ${actionName}.`);
      }
    }
  }

  console.log(`\nSummary: ${prepared} action(s) prepared, ${cached} already cached.`);
  console.log(`All actions ready in ${cacheDir} for offline runtime execution.\n`);
}

function cmdClean(args) {
  const cacheDir = resolveActionCacheDir(args.path);
  console.log('==================================================');
  console.log(' Morflow Action Cache Cleaner (Node.js CLI)');
  console.log(` Target Directory: ${cacheDir}`);
  console.log('==================================================');

  if (!fs.existsSync(cacheDir)) {
    console.log('Cache directory does not exist. Nothing to clean.');
    return;
  }

  let deletedCount = 0;
  for (const item of fs.readdirSync(cacheDir)) {
    const itemPath = path.join(cacheDir, item);
    fs.rmSync(itemPath, { recursive: true, force: true });
    deletedCount++;
  }

  console.log(`✓ Cleaned ${deletedCount} items from ${cacheDir}`);
}

function printHelp() {
  console.log(`Morflow - High-performance modular dataflow pipeline engine

Usage: morflow <COMMAND> [options]

Commands:
  prep <file>    Pre-downloads all actions required by a .morf pipeline ahead of time for offline execution
  clean          Cleans and removes all cached action binaries from the action path
  help           Print this help message

Options:
  --path <dir>           Custom action cache directory
  --repo <owner/repo>    GitHub repository (default: JiraPit/Morflow)
  --action-version <ver> Action Pack version tag (default: 0.1.0)
  --force                Force re-download even if already cached
  -h, --help             Show help
`);
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv.includes('-h') || argv.includes('--help') || argv[0] === 'help') {
    printHelp();
    return;
  }

  const command = argv[0];
  const args = {
    file: null,
    path: null,
    repo: 'JiraPit/Morflow',
    actionVersion: '0.1.0',
    force: false,
  };

  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--path' && i + 1 < argv.length) {
      args.path = argv[++i];
    } else if (arg === '--repo' && i + 1 < argv.length) {
      args.repo = argv[++i];
    } else if (arg === '--action-version' && i + 1 < argv.length) {
      args.actionVersion = argv[++i];
    } else if (arg === '--force') {
      args.force = true;
    } else if (!arg.startsWith('-') && !args.file) {
      args.file = arg;
    }
  }

  if (command === 'prep') {
    if (!args.file) {
      console.error('Error: Missing required argument <file> for prep command.\n');
      printHelp();
      process.exit(1);
    }
    await cmdPrep(args);
  } else if (command === 'clean') {
    cmdClean(args);
  } else {
    console.error(`Unknown command '${command}'.\n`);
    printHelp();
    process.exit(1);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
