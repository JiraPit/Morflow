#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const [archive] = process.argv.slice(2);
if (!archive) throw new Error('Usage: node scripts/publish-node.mjs <package.tgz>');
const metadata = JSON.parse(readFileSync(new URL('../bindings/js/package.json', import.meta.url)));
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const registry = 'https://registry.npmjs.org';
const query = spawnSync(npm, ['view', `${metadata.name}@${metadata.version}`, 'dist.integrity', '--json', '--registry', registry], { encoding: 'utf8' });
if (query.error) throw query.error;
if (query.status === 0) {
  const published = JSON.parse(query.stdout);
  const actual = `sha512-${createHash('sha512').update(readFileSync(archive)).digest('base64')}`;
  if (published !== actual) {
    throw new Error(`${metadata.name}@${metadata.version} is already published with different contents. Bump the version before publishing.`);
  }
  console.log(`${metadata.name}@${metadata.version} is already published with matching contents.`);
} else {
  let error;
  try { error = JSON.parse(query.stdout).error; } catch { /* Preserve the registry failure below. */ }
  if (error?.code !== 'E404') throw new Error(query.stderr || query.stdout || 'Cannot query npm registry.');
  const result = spawnSync(npm, ['publish', resolve(archive), '--access', 'public', '--registry', registry], { stdio: 'inherit' });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
}
