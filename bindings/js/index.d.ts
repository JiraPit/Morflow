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
  /**
   * The payload type to send. One of `'tensor'`, `'image'`, or `'audio'`.
   *
   * There is no type inference. A rank-3 array is a plain tensor unless you ask
   * for `'image'`, and a `[4, N]` float32 feature matrix is a tensor rather
   * than audio.
   *
   * @default 'tensor'
   */
  payloadType?: 'tensor' | 'image' | 'audio' | string;
  /**
   * Color space for `payloadType: 'image'`. Inferred from the channel count when omitted.
   */
  colorSpace?: 'grayscale' | 'rgb' | 'rgba' | 'bgr' | 'bgra' | string;
  /**
   * Sample rate in Hz for `payloadType: 'audio'`.
   *
   * @default 44100
   */
  sampleRate?: number;
  /**
   * Channel count for `payloadType: 'audio'`. Defaults to the leading dimension.
   */
  channels?: number;
  /**
   * Memory layout: `'hwc'` / `'chw'` for images, `'planar'` / `'interleaved'` for audio.
   *
   * @default 'hwc' for images, 'planar' for audio
   */
  layout?: 'hwc' | 'chw' | 'planar' | 'interleaved' | string;
}

export type PipelineInput = TensorInput | Float32Array | Buffer;

/**
 * A positional pipeline parameter: a tensor/audio/image object, a typed array,
 * a byte buffer, a plain number (Scalar or `*Arg`), a string (`StrArg`), or a
 * boolean (`BoolArg`), bound in declaration order.
 */
export type PipelineArg = TensorInput | Float32Array | Buffer | number | string | boolean;

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

  /**
   * Executes the pipeline synchronously with positional parameters bound in
   * declaration order. Parameters without a supplied value use their declared
   * default.
   */
  runSyncArgs(...args: PipelineArg[]): MorflowTensor;

  /**
   * Executes the pipeline synchronously with positional parameters, returning
   * all named emitted streams.
   */
  runSyncAllArgs(...args: PipelineArg[]): Record<string, MorflowTensor>;

  /**
   * Executes the pipeline asynchronously with positional parameters bound in
   * declaration order, returning a Promise.
   */
  runArgs(...args: PipelineArg[]): Promise<MorflowTensor>;

  /**
   * Executes the pipeline asynchronously with positional parameters, returning
   * a Promise with all named emitted streams.
   */
  runAllArgs(...args: PipelineArg[]): Promise<Record<string, MorflowTensor>>;
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

