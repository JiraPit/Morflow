#!/usr/bin/env node
// Puts Morflow's native libraries where an Android or iOS build picks them up:
//
//   node fetch-native.js android --app <app root> --out <dir>
//     <dir>/<abi>/libmorflow_capi.so and libmorflow_pack_<pack>.so, a jniLibs
//     directory for arm64-v8a and x86_64.
//   node fetch-native.js ios --app <app root> --out <dir>   (macOS only)
//     <dir>/morflow_capi.xcframework and morflow_pack_<pack>.xcframework, for
//     devices and the simulator.
//
// The packs are the ones the app's package.json lists under morflow.packs
// (written by `morflow prep --prod-target react-native`). Libraries come from
// Morflow's GitHub releases: the C interface from the engine release
// v<engineVersion>, each pack from action_packs/<pack>/v<version>. Every file is
// checked against its release's checksums.txt and cached, so later builds
// work offline. MORFLOW_RELEASE_URL (or morflow.releaseUrl in package.json)
// points at a mirror or a file: directory with the same layout.
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { pathToFileURL } = require('url');

const DEFAULT_RELEASE_URL = 'https://github.com/JiraPit/Morflow/releases/download';

const ANDROID_ABIS = { 'android-aarch64': 'arm64-v8a', 'android-x86_64': 'x86_64' };
const IOS_SLICES = { 'ios-arm64': 'iPhoneOS', 'ios-arm64-sim': 'iPhoneSimulator' };
// The engine and pack libraries are built for iOS 13 and later (scripts/native_targets.py).
const IOS_MINIMUM = '13.0';

class FetchError extends Error {}

/** The engine version this package runs, from src/version.ts. */
function engineVersion(packageRoot = path.join(__dirname, '..')) {
  const source = fs.readFileSync(path.join(packageRoot, 'src', 'version.ts'), 'utf8');
  const match = source.match(/engineVersion = '([^']+)'/);
  if (!match) throw new FetchError('Cannot read engineVersion from src/version.ts');
  return match[1];
}

/** The packs in package.json's morflow.packs: name -> exact version. */
function appPacks(packageJson) {
  const packs = (packageJson.morflow && packageJson.morflow.packs) || {};
  if (typeof packs !== 'object' || Array.isArray(packs)) {
    throw new FetchError('morflow.packs in package.json must map pack names to versions, such as {"audio_basics": "0.2.4"}');
  }
  for (const [name, version] of Object.entries(packs)) {
    if (!/^[a-z0-9_]+$/.test(name)) {
      throw new FetchError(`morflow.packs in package.json has an invalid pack name: ${name}`);
    }
    if (typeof version !== 'string' || !/^\d+\.\d+\.\d+$/.test(version)) {
      throw new FetchError(`morflow.packs.${name} in package.json must be an exact version such as "0.2.4"`);
    }
  }
  return packs;
}

const extension = (platform) => (platform.startsWith('ios-') ? 'dylib' : 'so');
const capiAsset = (platform) => `libmorflow_capi-${platform}.${extension(platform)}`;
const packAsset = (pack, version, platform) => `morflow_pack_${pack}-${version}-${platform}.${extension(platform)}`;
const engineTag = (version) => `v${version}`;
const packTag = (pack, version) => `action_packs/${pack}/v${version}`;

/** Every library the app needs: { stem, tag, version, asset(platform) }. */
function libraries(engine, packs) {
  return [
    { stem: 'morflow_capi', tag: engineTag(engine), version: engine, asset: capiAsset },
    ...Object.entries(packs)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([pack, version]) => ({
        stem: `morflow_pack_${pack}`,
        tag: packTag(pack, version),
        version,
        asset: (platform) => packAsset(pack, version, platform),
      })),
  ];
}

/** Parses a checksums.txt written by sha256sum. */
function parseChecksums(text) {
  const checksums = {};
  for (const line of text.split(/\r?\n/)) {
    const match = line.trim().match(/^([0-9a-f]{64}) [ *]?(.+)$/);
    if (match) checksums[match[2]] = match[1];
  }
  return checksums;
}

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const safe = (name) => name.replace(/[^A-Za-z0-9._-]/g, '_');

function defaultCache() {
  if (process.env.MORFLOW_CACHE) return process.env.MORFLOW_CACHE;
  if (process.platform === 'darwin') return path.join(os.homedir(), 'Library', 'Caches', 'morflow', 'releases');
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'morflow', 'releases');
}

/** Downloads release files into a cache, checked against checksums.txt. */
class Releases {
  constructor({ baseUrl = DEFAULT_RELEASE_URL, cache = defaultCache() } = {}) {
    // A directory may be given as a plain path.
    if (!/^[a-z][a-z0-9+.-]*:/i.test(baseUrl) || /^[a-z]:[\\/]/i.test(baseUrl)) {
      baseUrl = pathToFileURL(path.resolve(baseUrl)).href;
    }
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    this.cache = cache;
  }

  url(tag, asset) {
    return new URL(`${tag}/${asset}`, this.baseUrl);
  }

  async download(tag, asset, what) {
    const url = this.url(tag, asset);
    if (url.protocol === 'file:') {
      const file = require('url').fileURLToPath(url);
      if (!fs.existsSync(file)) throw new FetchError(`Could not find ${what}: ${file} does not exist.`);
      return fs.readFileSync(file);
    }
    let response;
    try {
      response = await fetch(url, { redirect: 'follow' });
    } catch (error) {
      throw new FetchError(`Could not download ${what} from ${url}: ${error.message}`);
    }
    if (response.status === 404) {
      throw new FetchError(`Could not find ${what}: ${url} returned 404. Is the version published?`);
    }
    if (!response.ok) {
      throw new FetchError(`Could not download ${what}: ${url} returned HTTP ${response.status}.`);
    }
    return Buffer.from(await response.arrayBuffer());
  }

  /** The cached file of `asset` from release `tag`, downloaded if needed. */
  async fetch(tag, asset) {
    const dir = path.join(this.cache, safe(tag));
    const file = path.join(dir, safe(asset));
    const checksumsFile = path.join(dir, 'checksums.txt');
    // Published files never change, so a cached checksum stays valid.
    if (fs.existsSync(checksumsFile) && fs.existsSync(file)) {
      const expected = parseChecksums(fs.readFileSync(checksumsFile, 'utf8'))[asset];
      if (expected && sha256(fs.readFileSync(file)) === expected) return file;
    }
    const checksumsText = (await this.download(tag, 'checksums.txt', `release ${tag}`)).toString('utf8');
    const expected = parseChecksums(checksumsText)[asset];
    if (!expected) throw new FetchError(`Morflow release ${tag} has no ${asset} (${this.url(tag, '')}).`);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(checksumsFile, checksumsText);
    const bytes = await this.download(tag, asset, `${asset} from release ${tag}`);
    const actual = sha256(bytes);
    if (actual !== expected) {
      throw new FetchError(
        `${asset} from Morflow release ${tag} does not match its checksum (expected ${expected}, got ${actual}). ` +
          'The download was corrupted or changed; it was not used.'
      );
    }
    const partial = `${file}.partial`;
    fs.writeFileSync(partial, bytes);
    fs.renameSync(partial, file);
    return file;
  }
}

/** Copies `from` to `to` unless `to` already has the same bytes. */
function place(from, to) {
  const bytes = fs.readFileSync(from);
  if (fs.existsSync(to) && sha256(fs.readFileSync(to)) === sha256(bytes)) return;
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.writeFileSync(to, bytes);
}

/** Removes entries of `dir` that are not in `keep`. */
function prune(dir, keep) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir)) {
    if (!keep.has(entry)) fs.rmSync(path.join(dir, entry), { recursive: true, force: true });
  }
}

async function android(releases, libs, out) {
  for (const [platform, abi] of Object.entries(ANDROID_ABIS)) {
    const keep = new Set();
    for (const lib of libs) {
      const file = `lib${lib.stem}.so`;
      place(await releases.fetch(lib.tag, lib.asset(platform)), path.join(out, abi, file));
      keep.add(file);
    }
    prune(path.join(out, abi), keep);
  }
  prune(out, new Set(Object.values(ANDROID_ABIS)));
}

function infoPlist(stem, version, platform) {
  const identifier = `org.morflow.${stem.replace(/_/g, '-')}`;
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>${stem}</string>
  <key>CFBundleIdentifier</key><string>${identifier}</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>${stem}</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>${version}</string>
  <key>CFBundleVersion</key><string>${version}</string>
  <key>CFBundleSupportedPlatforms</key><array><string>${platform}</string></array>
  <key>MinimumOSVersion</key><string>${IOS_MINIMUM}</string>
</dict>
</plist>
`;
}

/**
 * Wraps each library into an xcframework: a framework per slice (devices and
 * the simulator), named like the engine looks for packs on iOS
 * (Frameworks/morflow_pack_<pack>.framework/morflow_pack_<pack>).
 */
async function ios(releases, libs, out) {
  if (process.platform !== 'darwin') throw new FetchError('iOS libraries can only be prepared on macOS');
  const run = (tool, args) => execFileSync(tool, args, { stdio: ['ignore', 'ignore', 'inherit'] });
  const keep = new Set();
  for (const lib of libs) {
    const name = `${lib.stem}.xcframework`;
    keep.add(name);
    const sources = {};
    for (const platform of Object.keys(IOS_SLICES)) {
      sources[platform] = await releases.fetch(lib.tag, lib.asset(platform));
    }
    // Rebuild only when a library changed.
    const stamp = path.join(out, `${lib.stem}.sha256`);
    const digest = sha256(Buffer.from(Object.values(sources).map((f) => sha256(fs.readFileSync(f))).join()));
    keep.add(`${lib.stem}.sha256`);
    if (fs.existsSync(path.join(out, name)) && fs.existsSync(stamp) && fs.readFileSync(stamp, 'utf8') === digest) {
      continue;
    }
    const work = fs.mkdtempSync(path.join(os.tmpdir(), 'morflow-'));
    try {
      const frameworks = [];
      for (const [platform, sdk] of Object.entries(IOS_SLICES)) {
        const framework = path.join(work, platform, `${lib.stem}.framework`);
        const binary = path.join(framework, lib.stem);
        fs.mkdirSync(framework, { recursive: true });
        fs.copyFileSync(sources[platform], binary);
        fs.writeFileSync(path.join(framework, 'Info.plist'), infoPlist(lib.stem, lib.version, sdk));
        run('install_name_tool', ['-id', `@rpath/${lib.stem}.framework/${lib.stem}`, binary]);
        // Changing the install name invalidates the signature; Xcode signs
        // again with the app's identity when it embeds the framework.
        run('codesign', ['--force', '--sign', '-', binary]);
        frameworks.push('-framework', framework);
      }
      fs.rmSync(path.join(out, name), { recursive: true, force: true });
      fs.mkdirSync(out, { recursive: true });
      run('xcodebuild', ['-create-xcframework', ...frameworks, '-output', path.join(out, name)]);
      fs.writeFileSync(stamp, digest);
    } finally {
      fs.rmSync(work, { recursive: true, force: true });
    }
  }
  prune(out, keep);
}

function parseArgs(argv) {
  const [target, ...rest] = argv;
  const options = {};
  for (let i = 0; i < rest.length; i += 2) {
    if (!rest[i].startsWith('--') || rest[i + 1] === undefined) {
      throw new FetchError(`Unexpected argument ${rest[i]}`);
    }
    options[rest[i].slice(2)] = rest[i + 1];
  }
  if (!['android', 'ios'].includes(target) || !options.app || !options.out) {
    throw new FetchError('Usage: fetch-native.js <android|ios> --app <app root> --out <dir>');
  }
  return { target, app: options.app, out: options.out };
}

async function main(argv) {
  const { target, app, out } = parseArgs(argv);
  const packageJsonPath = path.join(app, 'package.json');
  if (!fs.existsSync(packageJsonPath)) {
    throw new FetchError(`There is no package.json in ${app}, the React Native app's root directory`);
  }
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const packs = appPacks(packageJson);
  const releases = new Releases({
    baseUrl: process.env.MORFLOW_RELEASE_URL || (packageJson.morflow && packageJson.morflow.releaseUrl) || undefined,
  });
  const libs = libraries(engineVersion(), packs);
  await (target === 'android' ? android : ios)(releases, libs, out);
  const listed = Object.entries(packs).map(([p, v]) => `${p} ${v}`);
  console.log(`Morflow: engine ${engineVersion()}, packs: ${listed.length ? listed.join(', ') : 'none'}`);
}

if (require.main === module) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error instanceof FetchError ? `Morflow: ${error.message}` : error);
    process.exit(1);
  });
}

module.exports = {
  FetchError,
  Releases,
  appPacks,
  engineVersion,
  infoPlist,
  libraries,
  parseArgs,
  parseChecksums,
  android,
  main,
};
