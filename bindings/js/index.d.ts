/**
 * Morflow - High-Performance Dataflow Pipeline Engine for JavaScript & TypeScript.
 */

/**
 * Every tensor element type.
 */
export type TensorDType = 'f32' | 'f64' | 'u8' | 'i8' | 'i16' | 'i32' | 'i64' | 'u32' | 'u64';

export interface MorflowTensor {
  shape: number[];
  dtype: TensorDType | string;
  data: Buffer;
  readonly rank: number;
  readonly length: number;
  /**
   * The elements as the typed array matching `dtype` (`BigInt64Array` for
   * `'i64'`, `BigUint64Array` for `'u64'`).
   */
  toTypedArray(): ArrayBufferView;
  toFloat32Array(): Float32Array;
  toFloat64Array(): Float64Array;
  toUint8Array(): Uint8Array;
  toInt8Array(): Int8Array;
  toInt16Array(): Int16Array;
  toInt32Array(): Int32Array;
  toUint32Array(): Uint32Array;
  toBigInt64Array(): BigInt64Array;
  toBigUint64Array(): BigUint64Array;
  toBuffer(): Buffer;
}

/**
 * An emitted value: a tensor, or an array of values for a Composite.
 */
export type MorflowOutput = MorflowTensor | MorflowOutput[];

export interface TensorInput {
  data: Buffer | ArrayBufferView | ArrayBuffer;
  shape: number[];
  /**
   * Element type of `data`. Inferred from a typed array other than
   * `Uint8Array`; otherwise `'f32'`.
   */
  dtype?: TensorDType | string;
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

/**
 * A typed array other than `Uint8Array` is a 1-D tensor of its own dtype
 * (`Float64Array` is `'f64'`, `BigInt64Array` is `'i64'`, …).
 */
export type TypedTensorArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

/**
 * A pipeline input: a tensor/audio/image object, a typed array, a byte buffer,
 * or an array of tensor inputs, which is sent as a Composite (so `[a, b]`
 * feeds the two-tensor form of add/sub/mul/div/pow).
 */
export type PipelineInput = TensorInput | TypedTensorArray | Buffer | CompositeInput;

/**
 * The components of a Composite, in order.
 */
export type CompositeInput = Array<TensorInput | TypedTensorArray | CompositeInput>;

/**
 * A positional pipeline parameter: a tensor/audio/image object, a typed array,
 * a byte buffer, an array (Composite), a plain number (Scalar or `*Arg`), a
 * string (`StrArg`), or a boolean (`BoolArg`), bound in declaration order.
 */
export type PipelineArg = PipelineInput | number | string | boolean;

/**
 * A compiled pipeline. Each run method returns `MorflowTensor` by default; a
 * Composite output is an array, so name its type, e.g. `run<MorflowTensor[]>(…)`.
 */
export class Pipeline {
  readonly params: string[];

  /**
   * Executes the pipeline synchronously on the current thread.
   */
  runSync<T extends MorflowOutput = MorflowTensor>(input?: PipelineInput): T;

  /**
   * Executes the pipeline synchronously, returning all named emitted streams
   * as an object keyed by output name.
   */
  runSyncToMap<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(input?: PipelineInput): T;

  /**
   * Executes the pipeline asynchronously on a worker thread pool, returning a Promise.
   * This offloads heavy computation and prevents blocking the Node.js event loop.
   */
  run<T extends MorflowOutput = MorflowTensor>(input?: PipelineInput): Promise<T>;

  /**
   * Executes the pipeline asynchronously, returning a Promise with all named
   * emitted streams as an object keyed by output name.
   */
  runToMap<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(input?: PipelineInput): Promise<T>;

  /**
   * Executes the pipeline synchronously with positional parameters bound in
   * declaration order. Parameters without a supplied value use their declared
   * default.
   */
  runSyncArgs<T extends MorflowOutput = MorflowTensor>(...args: PipelineArg[]): T;

  /**
   * Executes the pipeline synchronously with positional parameters, returning
   * all named emitted streams as an object keyed by output name.
   */
  runSyncToMapArgs<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(...args: PipelineArg[]): T;

  /**
   * Executes the pipeline asynchronously with positional parameters bound in
   * declaration order, returning a Promise.
   */
  runArgs<T extends MorflowOutput = MorflowTensor>(...args: PipelineArg[]): Promise<T>;

  /**
   * Executes the pipeline asynchronously with positional parameters, returning
   * a Promise with all named emitted streams as an object keyed by output name.
   */
  runToMapArgs<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(...args: PipelineArg[]): Promise<T>;

  /** @deprecated Renamed to `runSyncToMap`. */
  runSyncAll<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(input?: PipelineInput): T;
  /** @deprecated Renamed to `runToMap`. */
  runAll<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(input?: PipelineInput): Promise<T>;
  /** @deprecated Renamed to `runSyncToMapArgs`. */
  runSyncAllArgs<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(...args: PipelineArg[]): T;
  /** @deprecated Renamed to `runToMapArgs`. */
  runAllArgs<T extends Record<string, MorflowOutput> = Record<string, MorflowTensor>>(...args: PipelineArg[]): Promise<T>;
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

