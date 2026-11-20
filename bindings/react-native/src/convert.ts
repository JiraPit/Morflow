// Converting between JavaScript values and what the native module takes and
// returns (see cpp/MorflowHost.h). Kept apart from the native module so it can
// be tested without React Native.
import type { MorflowOutput, MorflowTensor, TensorInput } from './types';

/** An output as cpp/MorflowHost.cpp returns it. */
interface NativeOutput {
  dtype: string;
  shape?: number[];
  data?: ArrayBuffer;
  items?: NativeOutput[];
}

// ---------------------------------------------------------------------------
// Inputs

const DTYPE_ARRAYS = {
  f32: Float32Array,
  f64: Float64Array,
  u8: Uint8Array,
  i8: Int8Array,
  i16: Int16Array,
  i32: Int32Array,
  i64: BigInt64Array,
  u32: Uint32Array,
  u64: BigUint64Array,
} as const;

/**
 * The dtype a typed array carries. A Uint8Array has none: bare it is raw
 * bytes, and as tensor data it keeps the 'f32' default.
 */
function typedArrayDtype(value: unknown): string | undefined {
  for (const [dtype, Ctor] of Object.entries(DTYPE_ARRAYS)) {
    if (dtype !== 'u8' && value instanceof Ctor) return dtype;
  }
  return undefined;
}

function bytes(data: ArrayBufferView | ArrayBuffer): Uint8Array {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new TypeError('Tensor data must be a typed array or an ArrayBuffer');
}

/**
 * A host value as the native module takes it: an array becomes a Composite,
 * a typed array a 1-D tensor of its own dtype, and a Uint8Array raw bytes.
 */
export function toNative(value: unknown): unknown {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) {
    return { kind: 'tensor', payloadType: 'composite', items: value.map(toNativeComponent) };
  }
  if (ArrayBuffer.isView(value)) {
    const dtype = typedArrayDtype(value);
    return dtype
      ? { kind: 'tensor', data: bytes(value), shape: [(value as unknown as { length: number }).length], dtype }
      : { kind: 'bytes', data: bytes(value) };
  }
  if (value instanceof ArrayBuffer) {
    return { kind: 'bytes', data: new Uint8Array(value) };
  }
  if (typeof value === 'object' && (value as TensorInput).data !== undefined) {
    const input = value as TensorInput;
    if (!Array.isArray(input.shape)) {
      throw new TypeError('A tensor input needs a shape array');
    }
    return {
      kind: 'tensor',
      data: bytes(input.data),
      shape: input.shape,
      dtype: input.dtype ?? typedArrayDtype(input.data),
      payloadType: input.payloadType,
      colorSpace: input.colorSpace,
      layout: input.layout,
      sampleRate: input.sampleRate,
      channels: input.channels,
    };
  }
  throw new TypeError(
    'A pipeline input must be a tensor object, a typed array, a Uint8Array, an array, a number, a string or a boolean'
  );
}

function toNativeComponent(value: unknown): unknown {
  const converted = toNative(value) as { kind?: string } | undefined;
  if (!converted || typeof converted !== 'object' || converted.kind !== 'tensor') {
    throw new TypeError('Composite components must be tensor objects, typed arrays (not raw bytes), or arrays');
  }
  return converted;
}

// ---------------------------------------------------------------------------
// Outputs

/** A typed array over `data`, copying it first when its offset is not aligned. */
function typedView<T extends ArrayBufferView>(
  data: Uint8Array,
  Ctor: { new (buffer: ArrayBuffer, offset: number, length: number): T; BYTES_PER_ELEMENT: number }
): T {
  const source = data.byteOffset % Ctor.BYTES_PER_ELEMENT === 0 ? data : data.slice();
  return new Ctor(source.buffer as ArrayBuffer, source.byteOffset, source.byteLength / Ctor.BYTES_PER_ELEMENT);
}

/** A native output as a MorflowTensor, or an array of them for a Composite. */
export function wrapOutput(output: unknown): MorflowOutput | undefined {
  const value = output as NativeOutput | undefined;
  if (!value) return undefined;
  if (value.dtype === 'composite') {
    return (value.items ?? []).map((item) => wrapOutput(item) as MorflowOutput);
  }
  const tensor: MorflowTensor = {
    shape: value.shape ?? [],
    dtype: value.dtype,
    data: new Uint8Array(value.data ?? new ArrayBuffer(0)),
    get rank() {
      return this.shape.length;
    },
    get length() {
      return this.shape.reduce((a, b) => a * b, 1);
    },
    toTypedArray() {
      const Ctor = DTYPE_ARRAYS[this.dtype as keyof typeof DTYPE_ARRAYS];
      if (!Ctor) throw new TypeError(`No typed array for dtype '${this.dtype}'`);
      return typedView(this.data, Ctor as never);
    },
    toFloat32Array: () => typedView(tensor.data, Float32Array),
    toFloat64Array: () => typedView(tensor.data, Float64Array),
    toUint8Array: () => tensor.data,
    toInt8Array: () => typedView(tensor.data, Int8Array),
    toInt16Array: () => typedView(tensor.data, Int16Array),
    toInt32Array: () => typedView(tensor.data, Int32Array),
    toUint32Array: () => typedView(tensor.data, Uint32Array),
    toBigInt64Array: () => typedView(tensor.data, BigInt64Array),
    toBigUint64Array: () => typedView(tensor.data, BigUint64Array),
    toBuffer: () => tensor.data,
  };
  return tensor;
}

export function wrapOutputs(outputs: unknown): Record<string, MorflowOutput> {
  const result: Record<string, MorflowOutput> = {};
  for (const [name, value] of Object.entries(outputs as Record<string, unknown>)) {
    result[name] = wrapOutput(value) as MorflowOutput;
  }
  return result;
}
