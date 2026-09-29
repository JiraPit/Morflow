#!/usr/bin/env node
/**
 * Morflow Node.js CLI Entrypoint.
 * Delegates command execution directly to the native Morflow Rust CLI implementation.
 */

import process from 'node:process';
import morflow from '../index.js';

const exitCode = morflow.runCli(process.argv.slice(2));
process.exit(exitCode);
