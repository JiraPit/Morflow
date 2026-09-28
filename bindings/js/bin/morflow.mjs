#!/usr/bin/env node
/**
 * Morflow Node.js CLI
 * Provides 'prep', 'clean', 'spec', 'search', and 'list' subcommands when installed via npm or run via npx.
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import process from 'node:process';

const KNOWN_ACTIONS = [
  ['base', 'identity'],
  ['base', 'to_tensor'],
  ['audio_essentials', 'to_audio'],
  ['audio_essentials', 'to_pcm'],
  ['audio_essentials', 'to_wav'],
  ['audio_essentials', 'gain'],
  ['audio_essentials', 'normalize'],
  ['audio_essentials', 'biquad_filter'],
  ['audio_essentials', 'compressor'],
  ['audio_essentials', 'limiter'],
  ['audio_essentials', 'noise_gate'],
  ['audio_essentials', 'stereo_widen'],
  ['audio_essentials', 'resample'],
  ['audio_essentials', 'stft'],
  ['audio_essentials', 'delay'],
  ['image_essentials', 'to_image'],
  ['image_essentials', 'resize'],
  ['image_essentials', 'crop'],
  ['image_essentials', 'pad'],
  ['image_essentials', 'color_adjust'],
  ['image_essentials', 'gaussian_blur'],
  ['image_essentials', 'edge_detect'],
  ['image_essentials', 'sharpen'],
  ['image_essentials', 'threshold'],
  ['image_essentials', 'rotate'],
  ['image_essentials', 'flip'],
  ['image_essentials', 'blend'],
  ['image_essentials', 'morphology'],
];

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

function normalizePackName(pack) {
  const p = pack.trim().toLowerCase();
  if (p === 'audio_essential' || p === 'audio_essentials') {
    return 'audio_essentials';
  } else if (p === 'image_essential' || p === 'image_essentials') {
    return 'image_essentials';
  } else if (p === 'base') {
    return 'base';
  }
  return p;
}

function parseTargetPath(pathStr) {
  const clean = pathStr.replace(/::/g, '/').trim();
  let parts;
  if (clean.includes('/')) {
    parts = clean.split('/').map(s => s.trim()).filter(Boolean);
  } else if (clean.includes('.')) {
    parts = clean.split('.').map(s => s.trim()).filter(Boolean);
  } else {
    parts = clean ? [clean] : [];
  }

  if (parts.length >= 3) {
    const pack = normalizePackName(parts[0]);
    const action = parts[parts.length - 1];
    const rawVersion = parts.slice(1, parts.length - 1).join('/');
    const version = rawVersion.replace(/^v/, '');
    return [pack, version, action];
  } else if (parts.length === 2) {
    const pack = normalizePackName(parts[0]);
    const version = parts[1].replace(/^v/, '');
    return [pack, version, null];
  }

  const hintPack = parts.length > 0 ? parts[0] : 'base';
  const hintAct = parts.length > 0 ? parts[parts.length - 1] : 'identity';
  throw new Error(
    `Invalid target '${pathStr}'. Expected full action path '<package>/<version>/<action>' (e.g. '${hintPack}/latest/${hintAct}') or package path '<package>/<version>' (e.g. '${hintPack}/latest' or '${hintPack}/0.1.0').`
  );
}

function parseFullActionPath(pathStr) {
  const [pack, version, action] = parseTargetPath(pathStr);
  if (!action) {
    throw new Error(`Action name required in path '${pathStr}'.`);
  }
  return [pack, version, action];
}

function extractActionsFromMorf(source) {
  const importedSymbols = new Map();
  const importedPackages = new Map();
  const actions = [];

  for (const rawLine of source.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#') || line.startsWith('//')) continue;

    // 1. from <pkg>/<ver> import <item1> [as <alias1>], ...
    const fromMatch = line.match(/^from\s+([a-zA-Z0-9_]+)[/.][^\s]+\s+import\s+(.+)$/);
    if (fromMatch) {
      const pack = normalizePackName(fromMatch[1]);
      const itemsStr = fromMatch[2];
      for (const rawItem of itemsStr.split(',')) {
        const item = rawItem.trim();
        if (!item) continue;
        if (item.includes(' as ')) {
          const [realName, alias] = item.split(' as ').map(s => s.trim());
          importedSymbols.set(alias, [pack, realName]);
        } else {
          importedSymbols.set(item, [pack, item]);
        }
      }
      continue;
    }

    // 2. import <pkg>/<ver>/<action> [as <alias>]
    const singleMatch = line.match(/^import\s+([a-zA-Z0-9_]+)\/([^/\s]+)\/([a-zA-Z0-9_]+)(?:\s+as\s+([a-zA-Z0-9_]+))?/);
    if (singleMatch) {
      const pack = normalizePackName(singleMatch[1]);
      const actName = singleMatch[3];
      const alias = singleMatch[4] || actName;
      importedSymbols.set(alias, [pack, actName]);
      continue;
    }

    // 3. import <pkg>/<ver> [as <alias>] or import <pkg> [as <alias>]
    const importMatch = line.match(/^(?:import|use)\s+([a-zA-Z0-9_]+)(?:[/.]\S+)?(?:\s+as\s+([a-zA-Z0-9_]+))?/);
    if (importMatch) {
      const pack = normalizePackName(importMatch[1]);
      const alias = importMatch[2] || pack;
      importedPackages.set(alias, pack);
      continue;
    }

    // Match action calls in chain: >> action_name(...) or >> action_name
    const actionRegex = />>\s*([a-zA-Z0-9_/.:]+)/g;
    let match;
    while ((match = actionRegex.exec(line)) !== null) {
      const actFull = match[1].split('(')[0].trim();
      if (!actFull.startsWith('$') && !['emit', 'resurface', 'each', 'if', 'route', 'else'].includes(actFull) && !actions.includes(actFull)) {
        actions.push(actFull);
      }
    }
  }

  return actions.map(act => {
    if (importedSymbols.has(act)) {
      return importedSymbols.get(act);
    }
    const clean = act.replace(/::/g, '/');
    let parts;
    if (clean.includes('/')) {
      parts = clean.split('/').map(s => s.trim()).filter(Boolean);
    } else if (clean.includes('.')) {
      parts = clean.split('.').map(s => s.trim()).filter(Boolean);
    } else {
      parts = [clean];
    }

    if (parts.length >= 3) {
      const pack = importedPackages.get(parts[0]) || normalizePackName(parts[0]);
      return [pack, parts[parts.length - 1]];
    } else if (parts.length === 2) {
      const pack = importedPackages.get(parts[0]) || normalizePackName(parts[0]);
      return [pack, parts[1]];
    } else {
      for (const [kPack, kAct] of KNOWN_ACTIONS) {
        if (kAct === act) return [kPack, act];
      }
      if (importedPackages.size === 1) {
        return [Array.from(importedPackages.values())[0], act];
      }
      return ['base', act];
    }
  });
}

const latestVersionCache = new Map();

function resolveRepo() {
  return process.env.MORFLOW_REPO || 'JiraPit/Morflow';
}

async function fetchLatestPackVersion(pack, repo) {
  const cacheKey = `${pack}:${repo}`;
  if (latestVersionCache.has(cacheKey)) {
    return latestVersionCache.get(cacheKey);
  }

  const url = `https://api.github.com/repos/${repo}/releases`;
  const prefixV = `action_packs/${pack}/v`;
  const prefixNoV = `action_packs/${pack}/`;

  try {
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' },
      signal: AbortSignal.timeout(5000),
    });
    if (resp.ok) {
      const releases = await resp.json();
      if (Array.isArray(releases)) {
        for (const rel of releases) {
          const tag = rel.tag_name || '';
          if (tag.startsWith(prefixV)) {
            const ver = tag.slice(prefixV.length);
            if (ver) {
              latestVersionCache.set(cacheKey, ver);
              return ver;
            }
          } else if (tag.startsWith(prefixNoV)) {
            const ver = tag.slice(prefixNoV.length).replace(/^v/, '');
            if (ver) {
              latestVersionCache.set(cacheKey, ver);
              return ver;
            }
          }
        }
      }
    }
  } catch {}

  latestVersionCache.set(cacheKey, '0.1.0');
  return '0.1.0';
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

  const repo = resolveRepo();
  const [platformName, ext] = getHostPlatform();
  const cacheDir = resolveActionCacheDir(args.path);
  fs.mkdirSync(cacheDir, { recursive: true });

  console.log('==================================================');
  console.log(' Morflow Action Pre-Downloader (Node.js CLI)');
  console.log(` Pipeline: ${filePath}`);
  console.log(` Host Platform: ${platformName} (.${ext})`);
  console.log(` Cache Directory: ${cacheDir}`);
  console.log(` Repository: ${repo}`);
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

    const actionVersion = await fetchLatestPackVersion(pack, repo);
    console.log(`  [↓ Downloading] [${pack}] ${actionName} v${actionVersion}...`);
    const binaryFilename = `${actionName}_action-${actionVersion}-${platformName}.${ext}`;

    const urls = [
      `https://github.com/${repo}/releases/download/action_packs%2F${pack}%2Fv${actionVersion}/${binaryFilename}`,
      `https://github.com/${repo}/releases/download/action_packs/${pack}/v${actionVersion}/${binaryFilename}`,
    ];

    let downloaded = false;
    for (const url of urls) {
      try {
        const resp = await fetch(url, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' } });
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
        path.resolve(`actions/${pack}/${actionName}/target/release/lib${actionName}.${ext}`),
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

async function cmdSpec(args) {
  let pack, pathVersion, actionName;
  try {
    [pack, pathVersion, actionName] = parseTargetPath(args.action);
  } catch (err) {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  }

  if (!actionName) {
    console.error(
      `Error: \`morflow spec\` requires a 3-term action path '<package>/<version>/<action>' (e.g. '${pack}/latest/<action>'). A package path has no specification.`
    );
    process.exit(1);
  }

  const repo = resolveRepo();
  const version = pathVersion !== 'latest' && pathVersion ? pathVersion : await fetchLatestPackVersion(pack, repo);

  // 1. Check local cache directory
  const cacheSpec = path.join(resolveActionCacheDir(), pack, actionName, 'SPEC.md');
  if (fs.existsSync(cacheSpec)) {
    try {
      const content = fs.readFileSync(cacheSpec, 'utf-8');
      process.stdout.write(content);
      return;
    } catch {}
  }

  // 2. Check local files in workspace
  const localCandidates = [
    path.resolve(`actions/${pack}/${actionName}/SPEC.md`),
    path.resolve(`../actions/${pack}/${actionName}/SPEC.md`),
  ];

  for (const cand of localCandidates) {
    if (fs.existsSync(cand)) {
      try {
        const content = fs.readFileSync(cand, 'utf-8');
        process.stdout.write(content);
        return;
      } catch {}
    }
  }

  // 3. Fetch version-specific SPEC.md from GitHub Release assets
  const releaseUrls = [
    `https://github.com/${repo}/releases/download/action_packs%2F${pack}%2Fv${version}/${actionName}_SPEC.md`,
    `https://github.com/${repo}/releases/download/action_packs/${pack}/v${version}/${actionName}_SPEC.md`,
  ];
  for (const url of releaseUrls) {
    try {
      const resp = await fetch(url, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' } });
      if (resp.ok) {
        const content = await resp.text();
        process.stdout.write(content);
        return;
      }
    } catch {}
  }

  // 4. Fallback: Fetch from Git Tag
  const tagUrl = `https://raw.githubusercontent.com/${repo}/action_packs/${pack}/v${version}/actions/${pack}/${actionName}/SPEC.md`;
  try {
    const resp = await fetch(tagUrl, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' } });
    if (resp.ok) {
      const content = await resp.text();
      process.stdout.write(content);
      return;
    }
  } catch {}

  // 5. Fallback: Fetch raw SPEC.md from GitHub main branch
  const mainUrl = `https://raw.githubusercontent.com/${repo}/main/actions/${pack}/${actionName}/SPEC.md`;
  try {
    const resp = await fetch(mainUrl, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' } });
    if (resp.ok) {
      const content = await resp.text();
      process.stdout.write(content);
      return;
    }
  } catch {}

  console.error(`Error: SPEC.md not found for action '${pack}/v${version}/${actionName}' (checked local paths, release assets, and ${mainUrl}).`);
  process.exit(1);
}

function cmdSearch(args) {
  const q = args.query.toLowerCase();
  const matches = [];
  for (const [pack, act] of KNOWN_ACTIONS) {
    const actLower = act.toLowerCase();
    const pos = actLower.indexOf(q);
    if (pos !== -1) {
      matches.push({ pack, act, pos, len: act.length });
    }
  }

  // Rank by: 1) earlier substring position, 2) shorter action length, 3) alphabetical
  matches.sort((a, b) => {
    if (a.pos !== b.pos) return a.pos - b.pos;
    if (a.len !== b.len) return a.len - b.len;
    if (a.act !== b.act) return a.act.localeCompare(b.act);
    return a.pack.localeCompare(b.pack);
  });

  const topMatches = matches.slice(0, args.limit);

  if (topMatches.length === 0) {
    console.log(`No matching actions found for query '${args.query}'.`);
  } else {
    for (const item of topMatches) {
      console.log(`${item.pack}/latest/${item.act}`);
    }
  }
}

function cmdList(args) {
  const cacheDir = resolveActionCacheDir(args.path);
  const [, ext] = getHostPlatform();
  const suffix = `_action.${ext}`;
  const foundActions = new Set();

  if (fs.existsSync(cacheDir)) {
    for (const item of fs.readdirSync(cacheDir)) {
      const itemPath = path.join(cacheDir, item);
      const stat = fs.statSync(itemPath);
      if (stat.isDirectory()) {
        const packName = item;
        for (const subItem of fs.readdirSync(itemPath)) {
          if (subItem.endsWith(suffix)) {
            const actName = subItem.slice(0, -suffix.length);
            foundActions.add(`${packName}/latest/${actName}`);
          }
        }
      } else if (stat.isFile() && item.endsWith(suffix)) {
        const actName = item.slice(0, -suffix.length);
        let packName = 'base';
        for (const [kPack, kAct] of KNOWN_ACTIONS) {
          if (kAct === actName) {
            packName = kPack;
            break;
          }
        }
        foundActions.add(`${packName}/latest/${actName}`);
      }
    }
  }

  const devTarget = path.resolve('target/release/actions');
  if (fs.existsSync(devTarget)) {
    for (const item of fs.readdirSync(devTarget)) {
      const itemPath = path.join(devTarget, item);
      const stat = fs.statSync(itemPath);
      if (stat.isDirectory()) {
        const packName = item;
        for (const subItem of fs.readdirSync(itemPath)) {
          if (subItem.endsWith(suffix)) {
            const actName = subItem.slice(0, -suffix.length);
            foundActions.add(`${packName}/latest/${actName}`);
          }
        }
      }
    }
  }

  const sortedActions = Array.from(foundActions).sort();
  if (sortedActions.length === 0) {
    console.log(`No installed actions found in ${cacheDir}.`);
    console.log("Run 'morflow prep <pipeline.morf>' to download required actions.");
  } else {
    for (const act of sortedActions) {
      console.log(act);
    }
  }
}

function printHelp() {
  console.log(`Morflow - High-performance modular dataflow pipeline engine

Usage: morflow <COMMAND> [options]

Commands:
  prep <file>     Pre-downloads all actions required by a .morf pipeline ahead of time for offline execution
  clean           Cleans and removes all cached action binaries from the action path
  spec <action>   Views the raw SPEC.md documentation for a specified action '<package>/<version>/<action>' (e.g. base/latest/identity)
  search <query>  Performs fuzzy search for actions by name and returns top matching full action paths
  list            Lists all action paths installed locally in the action cache
  install <path>  Installs a specific action binary into the local action cache based on full action path
  help            Print this help message

Options:
  --path <dir>           Custom action cache directory
  --limit <number>       Maximum search results to return (default: 5)
  --force                Force re-download even if already cached
  -h, --help             Show help
`);
}

async function cmdInstall(args) {
  let pack, pathVersion, actionName;
  try {
    [pack, pathVersion, actionName] = parseTargetPath(args.action);
  } catch (err) {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  }
  const repo = resolveRepo();
  const version = pathVersion !== 'latest' && pathVersion ? pathVersion : await fetchLatestPackVersion(pack, repo);

  const [platformName, ext] = getHostPlatform();
  const cacheDir = resolveActionCacheDir(args.path);
  const packDir = path.join(cacheDir, pack);
  fs.mkdirSync(packDir, { recursive: true });

  if (actionName) {
    // Install single action
    const targetFilePack = path.join(packDir, `${actionName}_action.${ext}`);
    const targetFileRoot = path.join(cacheDir, `${actionName}_action.${ext}`);

    if (!args.force && (fs.existsSync(targetFilePack) || fs.existsSync(targetFileRoot))) {
      console.log(`Action '${pack}/latest/${actionName}' is already installed in ${targetFilePack}. Use --force to reinstall.`);
      return;
    }

    console.log('==================================================');
    console.log(' Morflow Action Installer (Node.js CLI)');
    console.log(` Action: ${pack}/latest/${actionName}`);
    console.log(` Version: v${version}`);
    console.log(` Host Platform: ${platformName} (.${ext})`);
    console.log(` Cache Directory: ${cacheDir}`);
    console.log(` Repository: ${repo}`);
    console.log('==================================================');

    console.log(`  [↓ Downloading] [${pack}] ${actionName} v${version}...`);
    const binaryFilename = `${actionName}_action-${version}-${platformName}.${ext}`;

    const urls = [
      `https://github.com/${repo}/releases/download/action_packs%2F${pack}%2Fv${version}/${binaryFilename}`,
      `https://github.com/${repo}/releases/download/action_packs/${pack}/v${version}/${binaryFilename}`,
    ];

    let downloaded = false;
    for (const url of urls) {
      try {
        const resp = await fetch(url, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' } });
        if (resp.ok) {
          const buffer = Buffer.from(await resp.arrayBuffer());
          fs.writeFileSync(targetFilePack, buffer);
          fs.writeFileSync(targetFileRoot, buffer);
          console.log(`    ✓ Successfully installed to ${targetFilePack}`);
          downloaded = true;
          break;
        }
      } catch {}
    }

    if (!downloaded) {
      const localCandidates = [
        path.resolve(`target/release/actions/${pack}/${actionName}_action.${ext}`),
        path.resolve(`target/release/actions/${actionName}_action.${ext}`),
        path.resolve(`actions/${pack}/${actionName}/target/release/lib${actionName}.${ext}`),
      ];

      let copied = false;
      for (const cand of localCandidates) {
        if (fs.existsSync(cand)) {
          fs.copyFileSync(cand, targetFilePack);
          fs.copyFileSync(cand, targetFileRoot);
          console.log(`    ✓ Copied local build artifact from ${cand}`);
          copied = true;
          break;
        }
      }

      if (!copied) {
        console.error(`Error: Could not download remote binary or find local artifact for [${pack}] ${actionName}.`);
        process.exit(1);
      }
    }

    // Cache SPEC.md if available
    const specDir = path.join(cacheDir, pack, actionName);
    fs.mkdirSync(specDir, { recursive: true });
    const targetSpec = path.join(specDir, 'SPEC.md');
    const localSpec = path.resolve(`actions/${pack}/${actionName}/SPEC.md`);
    if (fs.existsSync(localSpec)) {
      try {
        fs.copyFileSync(localSpec, targetSpec);
      } catch {}
    }

    console.log(`\n✓ Installation complete: ${pack}/latest/${actionName} is ready for runtime use.\n`);
  } else {
    // Install full package
    const actions = KNOWN_ACTIONS.filter(([p]) => p === pack).map(([, a]) => a);
    if (actions.length === 0) {
      console.error(`Error: Unknown package '${pack}'.`);
      process.exit(1);
    }

    console.log('==================================================');
    console.log(' Morflow Action Pack Installer (Node.js CLI)');
    console.log(` Package: ${pack} (${actions.length} actions)`);
    console.log(` Version: v${version}`);
    console.log(` Host Platform: ${platformName} (.${ext})`);
    console.log(` Cache Directory: ${cacheDir}`);
    console.log(` Repository: ${repo}`);
    console.log('==================================================');

    let installedCount = 0;
    let cachedCount = 0;

    for (const act of actions) {
      const targetFilePack = path.join(packDir, `${act}_action.${ext}`);
      const targetFileRoot = path.join(cacheDir, `${act}_action.${ext}`);

      if (!args.force && (fs.existsSync(targetFilePack) || fs.existsSync(targetFileRoot))) {
        console.log(`  [✓ Cached] [${pack}] ${act}`);
        cachedCount += 1;
        continue;
      }

      console.log(`  [↓ Downloading] [${pack}] ${act} v${version}...`);
      const binaryFilename = `${act}_action-${version}-${platformName}.${ext}`;
      const urls = [
        `https://github.com/${repo}/releases/download/action_packs%2F${pack}%2Fv${version}/${binaryFilename}`,
        `https://github.com/${repo}/releases/download/action_packs/${pack}/v${version}/${binaryFilename}`,
      ];

      let downloaded = false;
      for (const url of urls) {
        try {
          const resp = await fetch(url, { headers: { 'User-Agent': 'Morflow-Node-CLI/0.1.2' } });
          if (resp.ok) {
            const buffer = Buffer.from(await resp.arrayBuffer());
            fs.writeFileSync(targetFilePack, buffer);
            fs.writeFileSync(targetFileRoot, buffer);
            console.log(`    ✓ Successfully installed to ${targetFilePack}`);
            downloaded = true;
            installedCount += 1;
            break;
          }
        } catch {}
      }

      if (!downloaded) {
        const localCandidates = [
          path.resolve(`target/release/actions/${pack}/${act}_action.${ext}`),
          path.resolve(`target/release/actions/${act}_action.${ext}`),
          path.resolve(`actions/${pack}/${act}/target/release/lib${act}.${ext}`),
        ];

        for (const cand of localCandidates) {
          if (fs.existsSync(cand)) {
            fs.copyFileSync(cand, targetFilePack);
            fs.copyFileSync(cand, targetFileRoot);
            console.log(`    ✓ Copied local build artifact from ${cand}`);
            installedCount += 1;
            break;
          }
        }
      }
    }

    console.log(`\nSummary: ${installedCount} action(s) installed/updated, ${cachedCount} already cached.`);
    console.log(`✓ Action package '${pack}/latest' is ready in ${cacheDir}.\n`);
  }
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
    action: null,
    query: null,
    path: null,
    limit: 5,
    force: false,
  };

  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--path' && i + 1 < argv.length) {
      args.path = argv[++i];
    } else if (arg === '--limit' && i + 1 < argv.length) {
      args.limit = parseInt(argv[++i], 10) || 5;
    } else if (arg === '--force') {
      args.force = true;
    } else if (!arg.startsWith('-')) {
      if (!args.file) args.file = arg;
      if (!args.action) args.action = arg;
      if (!args.query) args.query = arg;
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
  } else if (command === 'spec') {
    if (!args.action) {
      console.error('Error: Missing required argument <action> for spec command.\n');
      printHelp();
      process.exit(1);
    }
    await cmdSpec(args);
  } else if (command === 'search') {
    if (!args.query) {
      console.error('Error: Missing required argument <query> for search command.\n');
      printHelp();
      process.exit(1);
    }
    cmdSearch(args);
  } else if (command === 'list') {
    cmdList(args);
  } else if (command === 'install') {
    if (!args.action) {
      console.error('Error: Missing required argument <action> for install command.\n');
      printHelp();
      process.exit(1);
    }
    await cmdInstall(args);
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

