/**
 * Morflow for React Native: run `.morf` pipelines inside your app on Android
 * and iOS. The API matches the Node.js package (`morflow` on npm), with
 * `Uint8Array` where Node.js uses `Buffer`.
 */
/// <reference path="./morf.d.ts" />
import { toNative, wrapOutput, wrapOutputs } from './convert';
import type { MorflowOutput, MorflowTensor, PipelineArg, PipelineInput } from './types';
import { engineVersion } from './version';

export type * from './types';
export { engineVersion };

// ---------------------------------------------------------------------------
// The native module

/** What cpp/MorflowHost.cpp installs as `global.__morflow`. */
interface NativePipeline {
  readonly params: string[];
  run(all: boolean, input?: unknown): unknown;
  runArgs(all: boolean, args: unknown[]): unknown;
  runAsync(all: boolean, input?: unknown): Promise<unknown>;
  runArgsAsync(all: boolean, args: unknown[]): Promise<unknown>;
  close(): void;
}
interface NativeEngine {
  version(): string;
  fromStr(source: string): NativePipeline;
  load(path: string): NativePipeline;
}

declare const global: { __morflow?: NativeEngine };

declare function require(name: string): any;

let engine: NativeEngine | undefined;

function native(): NativeEngine {
  if (engine) return engine;
  if (!global.__morflow) {
    // Looked up on first use, so a missing native module reports itself here.
    let module: { install(): boolean } | null = null;
    try {
      module = require('./NativeMorflow').default;
    } catch {
      module = null;
    }
    if (!module) {
      throw new Error(
        'react-native-morflow is not in this build of the app. Rebuild it after installing the package ' +
          '(npx react-native run-android / run-ios); Expo Go cannot load it, so use a development build.'
      );
    }
    module.install();
  }
  const installed = global.__morflow;
  if (!installed) {
    throw new Error('react-native-morflow could not start its native module.');
  }
  const version = installed.version();
  if (version !== engineVersion) {
    throw new Error(
      `react-native-morflow needs engine ${engineVersion}, but the app bundles engine ${version}. ` +
        'Rebuild the app after updating react-native-morflow.'
    );
  }
  engine = installed;
  return engine;
}

// ---------------------------------------------------------------------------
// Pipelines

/**
 * A compiled pipeline. Each run method returns `MorflowTensor` by default; a
 * Composite output is an array, so name its type, e.g. `run<MorflowTensor[]>(…)`.
 */
export class Pipeline {
  /** @internal */
  constructor(private readonly native: NativePipeline) {}

  /** The pipeline's parameter names, in declaration order. */
  get params(): string[] {
    return this.native.params;
  }

  /**
   * Runs the pipeline on a worker thread, so the app stays responsive.
   * Resolves with the first emitted output.
   */
  async run<T extends MorflowOutput = MorflowTensor>(input?: PipelineInput): Promise<T> {
    return wrapOutput(await this.native.runAsync(false, toNative(input))) as T;
  }

  /** Like `run`, resolving with every named emitted output. */
  async runAll<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(
    input?: PipelineInput
  ): Promise<T> {
    return wrapOutputs(await this.native.runAsync(true, toNative(input))) as T;
  }

  /**
   * Runs the pipeline with positional parameters bound in declaration order.
   * Parameters without a supplied value use their declared default.
   */
  async runArgs<T extends MorflowOutput = MorflowTensor>(...args: PipelineArg[]): Promise<T> {
    return wrapOutput(await this.native.runArgsAsync(false, args.map(toNative))) as T;
  }

  /** Like `runArgs`, resolving with every named emitted output. */
  async runAllArgs<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(
    ...args: PipelineArg[]
  ): Promise<T> {
    return wrapOutputs(await this.native.runArgsAsync(true, args.map(toNative))) as T;
  }

  /** Runs the pipeline on the JavaScript thread. Prefer `run` in an app, which keeps the UI responsive. */
  runSync<T extends MorflowOutput = MorflowTensor>(input?: PipelineInput): T {
    return wrapOutput(this.native.run(false, toNative(input))) as T;
  }

  /** Like `runSync`, returning every named emitted output. */
  runSyncAll<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(input?: PipelineInput): T {
    return wrapOutputs(this.native.run(true, toNative(input))) as T;
  }

  /** `runArgs` on the JavaScript thread. */
  runSyncArgs<T extends MorflowOutput = MorflowTensor>(...args: PipelineArg[]): T {
    return wrapOutput(this.native.runArgs(false, args.map(toNative))) as T;
  }

  /** `runAllArgs` on the JavaScript thread. */
  runSyncAllArgs<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(
    ...args: PipelineArg[]
  ): T {
    return wrapOutputs(this.native.runArgs(true, args.map(toNative))) as T;
  }

  /**
   * Frees the pipeline now instead of when it is garbage collected. Runs
   * already started still finish.
   */
  close(): void {
    this.native.close();
  }
}

/**
 * Compiles a pipeline from its `.morf` source text, such as a pipeline
 * imported with the Metro helper:
 *
 *     import boost from './pipelines/boost.morf';
 *     const pipeline = fromStr(boost);
 */
export function fromStr(source: string): Pipeline {
  return new Pipeline(native().fromStr(source));
}

/** Alias for `fromStr`. */
export const fromString = fromStr;

/** Compiles a `.morf` pipeline file on the device, such as one the app downloaded. */
export function load(path: string): Pipeline {
  return new Pipeline(native().load(path));
}
