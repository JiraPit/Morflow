// The package's public types, shared with the Node.js package's API.

/** Every tensor element type. */
export type TensorDType = 'f32' | 'f64' | 'u8' | 'i8' | 'i16' | 'i32' | 'i64' | 'u32' | 'u64';

export interface MorflowTensor {
  shape: number[];
  dtype: TensorDType | string;
  /** The raw element bytes. */
  data: Uint8Array;
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
  /** The raw bytes, such as a WAV or PNG file from `to_wav` or `to_png`. */
  toBuffer(): Uint8Array;
}

/** An emitted value: a tensor, or an array of values for a Composite. */
export type MorflowOutput = MorflowTensor | MorflowOutput[];

export interface TensorInput {
  data: ArrayBufferView | ArrayBuffer;
  shape: number[];
  /**
   * Element type of `data`. Inferred from a typed array other than
   * `Uint8Array`; otherwise `'f32'`.
   */
  dtype?: TensorDType | string;
  /**
   * The payload type to send: `'tensor'`, `'image'` or `'audio'`. There is no
   * type inference: a rank-3 array is a plain tensor unless you ask for
   * `'image'`.
   *
   * @default 'tensor'
   */
  payloadType?: 'tensor' | 'image' | 'audio' | string;
  /** Color space for `payloadType: 'image'`. Inferred from the channel count when omitted. */
  colorSpace?: 'grayscale' | 'rgb' | 'rgba' | 'bgr' | 'bgra' | string;
  /**
   * Sample rate in Hz for `payloadType: 'audio'`.
   *
   * @default 44100
   */
  sampleRate?: number;
  /** Channel count for `payloadType: 'audio'`. Defaults to the leading dimension. */
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

/** The components of a Composite, in order. */
export type CompositeInput = Array<TensorInput | TypedTensorArray | CompositeInput>;

/**
 * A pipeline input: a tensor/audio/image object, a typed array, raw bytes (a
 * `Uint8Array`), or an array of tensor inputs, which is sent as a Composite.
 */
export type PipelineInput = TensorInput | TypedTensorArray | Uint8Array | CompositeInput;

/**
 * A positional pipeline parameter: a pipeline input, a plain number (Scalar or
 * `*Arg`), a string (`StrArg`), or a boolean (`BoolArg`), bound in
 * declaration order.
 */
export type PipelineArg = PipelineInput | number | string | boolean;
