#!/usr/bin/env node
import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const [artifacts, output] = process.argv.slice(2);
if (!artifacts || !output) {
  throw new Error('Usage: node scripts/package-node.mjs <native-artifacts> <output-directory>');
}
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const binding = join(root, 'bindings/js');
const destination = resolve(output);
const staging = join(destination, 'package');
const metadata = JSON.parse(readFileSync(join(binding, 'package.json'), 'utf8'));
const binaries = metadata.files.filter(name => name.endsWith('.node'));
// Check the complete platform set before creating a publishable package.
for (const name of binaries) {
  if (readFileSync(join(artifacts, name)).length === 0) {
    throw new Error(`Native addon is empty: ${name}`);
  }
}
mkdirSync(staging, { recursive: true });
copyFileSync(join(binding, 'package.json'), join(staging, 'package.json'));
for (const name of metadata.files) {
  const source = name.endsWith('.node') ? join(artifacts, name)
    : ['LICENSE', 'README.md'].includes(name) ? join(root, name) : join(binding, name);
  mkdirSync(dirname(join(staging, name)), { recursive: true });
  copyFileSync(source, join(staging, name));
}
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
execFileSync(npm, ['pack', '--ignore-scripts', '--pack-destination', destination], {
  cwd: staging, stdio: 'inherit'
});
