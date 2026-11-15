const binaries = {
  'linux-x64': 'morflow.linux-x64-gnu.node',
  'linux-arm64': 'morflow.linux-arm64-gnu.node',
  'darwin-arm64': 'morflow.darwin-arm64.node',
  'win32-x64': 'morflow.win32-x64-msvc.node'
};
const platform = `${process.platform}-${process.arch}`;
const binary = binaries[platform];
if (!binary) {
  throw new Error(`Morflow does not provide a native addon for ${platform}.`);
}
if (process.platform === 'linux' && !process.report.getReport().header.glibcVersionRuntime) {
  throw new Error('Morflow requires a glibc-based Linux system.');
}
const nativeBinding = require(`./${binary}`);

/**
 * Typed array class for each tensor dtype.
 */
const DTYPE_ARRAYS = {
  f32: Float32Array,
  f64: Float64Array,
  u8: Uint8Array,
  i8: Int8Array,
  i16: Int16Array,
  i32: Int32Array,
  i64: BigInt64Array,
  u32: Uint32Array,
  u64: BigUint64Array
};

/**
 * The dtype a typed array carries. A Uint8Array (and so a Buffer) has none:
 * bare it is raw bytes, and as tensor data it keeps the 'f32' default.
 */
function typedArrayDtype(value) {
  for (const [dtype, Ctor] of Object.entries(DTYPE_ARRAYS)) {
    if (dtype !== 'u8' && value instanceof Ctor) {
      return dtype;
    }
  }
  return undefined;
}

function toBuffer(data) {
  if (Buffer.isBuffer(data)) return data;
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Converts a host value into what the native addon accepts: an array becomes
 * a Composite with one component per item, and a typed array becomes a 1-D
 * tensor of its own dtype.
 */
function toNative(value) {
  if (Array.isArray(value)) {
    return {
      data: Buffer.alloc(0),
      shape: [],
      payloadType: 'composite',
      items: value.map(toNativeComponent)
    };
  }
  if (ArrayBuffer.isView(value)) {
    const dtype = typedArrayDtype(value);
    return dtype ? { data: toBuffer(value), shape: [value.length], dtype } : value;
  }
  if (value && typeof value === 'object' && value.data !== undefined) {
    return {
      ...value,
      data: toBuffer(value.data),
      dtype: value.dtype ?? typedArrayDtype(value.data)
    };
  }
  return value;
}

function toNativeComponent(value) {
  const native = toNative(value);
  if (!native || typeof native !== 'object' || ArrayBuffer.isView(native)) {
    throw new TypeError(
      'Composite components must be tensor objects, typed arrays (not raw byte buffers), or arrays'
    );
  }
  return native;
}

/**
 * A typed array over `buffer`, copying it first when its offset is not
 * aligned to the element size.
 */
function typedView(buffer, Ctor) {
  const source = buffer.byteOffset % Ctor.BYTES_PER_ELEMENT === 0 ? buffer : new Uint8Array(buffer);
  return new Ctor(source.buffer, source.byteOffset, source.byteLength / Ctor.BYTES_PER_ELEMENT);
}

/**
 * Enhances a MorflowTensor object with convenient typed array getters. A
 * Composite output becomes an array of its wrapped components.
 */
function wrapTensor(tensor) {
  if (tensor && tensor.dtype === 'composite') {
    return (tensor.items || []).map(wrapTensor);
  }
  if (!tensor || !tensor.data) return tensor;

  return {
    shape: tensor.shape,
    dtype: tensor.dtype,
    data: tensor.data,
    get rank() {
      return this.shape.length;
    },
    get length() {
      return this.shape.reduce((a, b) => a * b, 1);
    },
    /**
     * The elements as the typed array matching `dtype` (BigInt64Array for
     * 'i64', BigUint64Array for 'u64').
     */
    toTypedArray() {
      const Ctor = DTYPE_ARRAYS[this.dtype];
      if (!Ctor) {
        throw new TypeError(`No typed array for dtype '${this.dtype}'`);
      }
      return typedView(this.data, Ctor);
    },
    toFloat32Array() {
      return typedView(this.data, Float32Array);
    },
    toFloat64Array() {
      return typedView(this.data, Float64Array);
    },
    toUint8Array() {
      return typedView(this.data, Uint8Array);
    },
    toInt8Array() {
      return typedView(this.data, Int8Array);
    },
    toInt16Array() {
      return typedView(this.data, Int16Array);
    },
    toInt32Array() {
      return typedView(this.data, Int32Array);
    },
    toUint32Array() {
      return typedView(this.data, Uint32Array);
    },
    toBigInt64Array() {
      return typedView(this.data, BigInt64Array);
    },
    toBigUint64Array() {
      return typedView(this.data, BigUint64Array);
    },
    toBuffer() {
      return this.data;
    }
  };
}

function wrapOutputs(outputs) {
  const result = {};
  for (const key of Object.keys(outputs)) {
    result[key] = wrapTensor(outputs[key]);
  }
  return result;
}

/**
 * Enhanced Morflow Pipeline JavaScript wrapper.
 *
 * An input is a typed array (a 1-D tensor of its dtype), a Buffer (raw
 * bytes), a `{ data, shape, dtype?, payloadType? }` object, or an array of
 * those, which is sent as a Composite.
 */
class MorflowPipelineWrapper {
  constructor(nativePipeline) {
    this._native = nativePipeline;
  }

  get params() {
    return this._native.params;
  }

  /**
   * Executes the pipeline synchronously on the current thread.
   * @param {ArrayBufferView | Buffer | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string } | Array<any>} [input]
   * @returns {any}
   */
  runSync(input) {
    return wrapTensor(this._native.runSync(toNative(input)));
  }

  /**
   * Executes the pipeline synchronously returning a dictionary of all named streams.
   * @param {ArrayBufferView | Buffer | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string } | Array<any>} [input]
   * @returns {Record<string, any>}
   */
  runSyncAll(input) {
    return wrapOutputs(this._native.runSyncAll(toNative(input)));
  }

  /**
   * Executes the pipeline asynchronously on a worker thread, returning a Promise.
   * @param {ArrayBufferView | Buffer | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string } | Array<any>} [input]
   * @returns {Promise<any>}
   */
  async run(input) {
    return wrapTensor(await this._native.run(toNative(input)));
  }

  /**
   * Executes the pipeline asynchronously returning a Promise with all named streams.
   * @param {ArrayBufferView | Buffer | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string } | Array<any>} [input]
   * @returns {Promise<Record<string, any>>}
   */
  async runAll(input) {
    return wrapOutputs(await this._native.runAll(toNative(input)));
  }

  /**
   * Executes the pipeline synchronously with positional parameters bound in
   * declaration order. Each value is a tensor/audio/image object, a typed
   * array, a Buffer, an array (Composite), a plain number (Scalar or *Arg), a
   * string (StrArg), or a boolean (BoolArg).
   * @param {...(ArrayBufferView | Buffer | Array<any> | number | string | boolean | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {any}
   */
  runSyncArgs(...args) {
    return wrapTensor(this._native.runSyncArgs(args.map(toNative)));
  }

  /**
   * Executes the pipeline synchronously with positional parameters, returning
   * a dictionary of all named streams.
   * @param {...(ArrayBufferView | Buffer | Array<any> | number | string | boolean | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {Record<string, any>}
   */
  runSyncAllArgs(...args) {
    return wrapOutputs(this._native.runSyncAllArgs(args.map(toNative)));
  }

  /**
   * Executes the pipeline asynchronously with positional parameters, returning
   * a Promise.
   * @param {...(ArrayBufferView | Buffer | Array<any> | number | string | boolean | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {Promise<any>}
   */
  async runArgs(...args) {
    return wrapTensor(await this._native.runArgs(args.map(toNative)));
  }

  /**
   * Executes the pipeline asynchronously with positional parameters, returning
   * a Promise with all named streams.
   * @param {...(ArrayBufferView | Buffer | Array<any> | number | string | boolean | { data: ArrayBufferView | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {Promise<Record<string, any>>}
   */
  async runAllArgs(...args) {
    return wrapOutputs(await this._native.runAllArgs(args.map(toNative)));
  }
}

/**
 * Compiles and loads a .morf pipeline file from disk.
 * @param {string} path - Absolute or relative path to the .morf file.
 * @returns {MorflowPipelineWrapper}
 */
function load(path) {
  const native = nativeBinding.load(path);
  return new MorflowPipelineWrapper(native);
}

/**
 * Compiles a .morf pipeline DSL source string directly.
 * @param {string} source - Raw .morf DSL string.
 * @returns {MorflowPipelineWrapper}
 */
function fromStr(source) {
  const native = nativeBinding.fromStr(source);
  return new MorflowPipelineWrapper(native);
}

/**
 * Executes the Morflow CLI with the given array of argument strings.
 * @param {string[]} args - Command-line arguments.
 * @returns {number} Exit code (0 for success).
 */
function runCli(args) {
  return nativeBinding.runCli(args || []);
}

module.exports = {
  load,
  fromStr,
  fromString: fromStr,
  runCli,
  wrapTensor,
  Pipeline: MorflowPipelineWrapper,
  MorflowTensor: nativeBinding.MorflowTensor
};
