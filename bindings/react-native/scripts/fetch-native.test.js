'use strict';
// node --test scripts/

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const test = require('node:test');

const fetchNative = require('./fetch-native');

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

/** A release directory with `files` (asset -> content) and their checksums. */
function release(root, tag, files) {
  const dir = path.join(root, tag);
  fs.mkdirSync(dir, { recursive: true });
  let checksums = '';
  for (const [asset, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(dir, asset), content);
    checksums += `${sha256(Buffer.from(content))}  ${asset}\n`;
  }
  fs.writeFileSync(path.join(dir, 'checksums.txt'), checksums);
}

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'morflow-test-'));
}

test('reads the pack list and rejects what an app cannot bundle', () => {
  assert.deepEqual(fetchNative.appPacks({}), {});
  assert.deepEqual(fetchNative.appPacks({ morflow: { packs: { base: '0.3.4' } } }), { base: '0.3.4' });
  assert.throws(() => fetchNative.appPacks({ morflow: { packs: ['base'] } }), /must map pack names/);
  assert.throws(() => fetchNative.appPacks({ morflow: { packs: { 'Bad-Name': '1.0.0' } } }), /invalid pack name/);
  assert.throws(() => fetchNative.appPacks({ morflow: { packs: { base: 'latest' } } }), /exact version/);
});

test('names the release files the engine and packs are published as', () => {
  const [capi, pack] = fetchNative.libraries('0.3.4', { audio_basics: '0.2.4' });
  assert.equal(capi.tag, 'v0.3.4');
  assert.equal(capi.asset('android-aarch64'), 'libmorflow_capi-android-aarch64.so');
  assert.equal(capi.asset('ios-arm64-sim'), 'libmorflow_capi-ios-arm64-sim.dylib');
  assert.equal(pack.stem, 'morflow_pack_audio_basics');
  assert.equal(pack.tag, 'action_packs/audio_basics/v0.2.4');
  assert.equal(pack.asset('android-x86_64'), 'morflow_pack_audio_basics-0.2.4-android-x86_64.so');
});

test('the package, src/version.ts and the engine have one version', () => {
  const root = path.join(__dirname, '..');
  const engine = fs.readFileSync(path.join(root, '..', '..', 'pipeline', 'Cargo.toml'), 'utf8').match(/^version = "([^"]+)"/m)[1];
  assert.equal(fetchNative.engineVersion(), engine);
  assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version, engine);
});

test('lays out jniLibs for every Android ABI and removes stale libraries', async () => {
  const root = tempDir();
  const releases = path.join(root, 'releases');
  release(releases, 'v0.3.4', {
    'libmorflow_capi-android-aarch64.so': 'capi arm64',
    'libmorflow_capi-android-x86_64.so': 'capi x86_64',
  });
  release(releases, 'action_packs/base/v0.3.4', {
    'morflow_pack_base-0.3.4-android-aarch64.so': 'base arm64',
    'morflow_pack_base-0.3.4-android-x86_64.so': 'base x86_64',
  });
  const out = path.join(root, 'jniLibs');
  fs.mkdirSync(path.join(out, 'arm64-v8a'), { recursive: true });
  fs.writeFileSync(path.join(out, 'arm64-v8a', 'libmorflow_pack_old.so'), 'stale');
  fs.mkdirSync(path.join(out, 'armeabi-v7a'));

  const source = new fetchNative.Releases({ baseUrl: pathToFileURL(releases).href, cache: path.join(root, 'cache') });
  await fetchNative.android(source, fetchNative.libraries('0.3.4', { base: '0.3.4' }), out);

  assert.deepEqual(fs.readdirSync(out).sort(), ['arm64-v8a', 'x86_64']);
  assert.deepEqual(fs.readdirSync(path.join(out, 'arm64-v8a')).sort(), ['libmorflow_capi.so', 'libmorflow_pack_base.so']);
  assert.equal(fs.readFileSync(path.join(out, 'x86_64', 'libmorflow_pack_base.so'), 'utf8'), 'base x86_64');

  // A second run works from the cache, without the releases.
  fs.rmSync(releases, { recursive: true });
  await fetchNative.android(source, fetchNative.libraries('0.3.4', { base: '0.3.4' }), out);
  assert.equal(fs.readFileSync(path.join(out, 'arm64-v8a', 'libmorflow_capi.so'), 'utf8'), 'capi arm64');
});

test('refuses files that do not match their checksum or are not published', async () => {
  const root = tempDir();
  const releases = path.join(root, 'releases');
  release(releases, 'v0.3.4', { 'libmorflow_capi-android-aarch64.so': 'capi' });
  fs.writeFileSync(path.join(releases, 'v0.3.4', 'libmorflow_capi-android-aarch64.so'), 'tampered');
  const source = new fetchNative.Releases({ baseUrl: pathToFileURL(releases).href, cache: path.join(root, 'cache') });
  await assert.rejects(source.fetch('v0.3.4', 'libmorflow_capi-android-aarch64.so'), /does not match its checksum/);
  await assert.rejects(source.fetch('v0.3.4', 'libmorflow_capi-android-x86_64.so'), /has no libmorflow_capi-android-x86_64.so/);
  await assert.rejects(source.fetch('v9.9.9', 'x.so'), /Could not find release v9.9.9/);
});

test('a release directory can be a plain path', async () => {
  const root = tempDir();
  release(path.join(root, 'releases'), 'v0.3.4', { 'libmorflow_capi-android-aarch64.so': 'capi' });
  const source = new fetchNative.Releases({ baseUrl: path.join(root, 'releases'), cache: path.join(root, 'cache') });
  const file = await source.fetch('v0.3.4', 'libmorflow_capi-android-aarch64.so');
  assert.equal(fs.readFileSync(file, 'utf8'), 'capi');
});

test('framework Info.plist names the binary and the platform', () => {
  const plist = fetchNative.infoPlist('morflow_pack_base', '0.3.4', 'iPhoneSimulator');
  assert.match(plist, /<key>CFBundleExecutable<\/key><string>morflow_pack_base<\/string>/);
  assert.match(plist, /<string>org\.morflow\.morflow-pack-base<\/string>/);
  assert.match(plist, /<string>iPhoneSimulator<\/string>/);
  assert.match(plist, /<key>MinimumOSVersion<\/key><string>13\.0<\/string>/);
});

test('command line', () => {
  assert.deepEqual(fetchNative.parseArgs(['android', '--app', 'a', '--out', 'b']), { target: 'android', app: 'a', out: 'b' });
  assert.throws(() => fetchNative.parseArgs(['web', '--app', 'a', '--out', 'b']), /Usage/);
  assert.throws(() => fetchNative.parseArgs(['android', '--app']), /Unexpected argument/);
});
