/**
 * Morflow - High-Performance Dataflow Pipeline Engine for JavaScript & TypeScript.
 */

export interface MorflowTensor {
  shape: number[];
  dtype: 'f32' | 'u8' | 'i32' | string;
  data: Buffer;
  readonly rank: number;
  readonly length: number;
  toFloat32Array(): Float32Array;
  toUint8Array(): Uint8Array;
  toInt32Array(): Int32Array;
  toBuffer(): Buffer;
}

export interface TensorInput {
  data: Buffer | Uint8Array | Float32Array;
  shape: number[];
  dtype?: 'f32' | 'u8' | 'i32' | string;
}

export type PipelineInput =
  | TensorInput
  | Float32Array
  | Uint8Array
  | Buffer
  | string
  | number
  | boolean;

export class Pipeline {
  readonly params: string[];

  /**
   * Executes the pipeline synchronously on the current thread.
   */
  runSync(input?: PipelineInput): MorflowTensor;

  /**
   * Executes the pipeline synchronously, returning all named emitted streams.
   */
  runSyncAll(input?: PipelineInput): Record<string, MorflowTensor>;

  /**
   * Executes the pipeline asynchronously on a worker thread pool, returning a Promise.
   * This offloads heavy computation and prevents blocking the Node.js event loop.
   */
  run(input?: PipelineInput): Promise<MorflowTensor>;

  /**
   * Executes the pipeline asynchronously, returning all named emitted streams as a Promise.
   */
  runAll(input?: PipelineInput): Promise<Record<string, MorflowTensor>>;
}

/**
 * Compiles and loads a `.morf` pipeline file from disk.
 * @param path - File path to the `.morf` pipeline definition.
 */
export function load(path: string): Pipeline;

/**
 * Compiles a `.morf` pipeline DSL source string directly.
 * @param source - Raw `.morf` DSL text content.
 */
export function fromStr(source: string): Pipeline;

/**
 * Alias for `fromStr`.
 */
export function fromString(source: string): Pipeline;

/**
 * Executes the Morflow CLI with the given array of argument strings.
 * @param args - Command-line arguments.
 * @returns Exit code (0 for success).
 */
export function runCli(args: string[]): number;

