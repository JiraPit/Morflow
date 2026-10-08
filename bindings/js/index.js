const binaries = {
  'linux-x64': 'morflow.linux-x64-gnu.node',
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
 * Enhances a MorflowTensor object with convenient typed array getters.
 */
function wrapTensor(tensor) {
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
    toFloat32Array() {
      return new Float32Array(
        this.data.buffer,
        this.data.byteOffset,
        this.data.byteLength / 4
      );
    },
    toUint8Array() {
      return new Uint8Array(
        this.data.buffer,
        this.data.byteOffset,
        this.data.byteLength
      );
    },
    toInt32Array() {
      return new Int32Array(
        this.data.buffer,
        this.data.byteOffset,
        this.data.byteLength / 4
      );
    },
    toBuffer() {
      return this.data;
    }
  };
}

/**
 * Enhanced Morflow Pipeline JavaScript wrapper.
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
   * @param {Float32Array | Uint8Array | Buffer | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string }} [input]
   * @returns {any}
   */
  runSync(input) {
    const res = this._native.runSync(input);
    if (res && res.shape && res.data) {
      return wrapTensor(res);
    }
    return res;
  }

  /**
   * Executes the pipeline synchronously returning a dictionary of all named streams.
   * @param {Float32Array | Uint8Array | Buffer | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string }} [input]
   * @returns {Record<string, any>}
   */
  runSyncAll(input) {
    const outputs = this._native.runSyncAll(input);
    const result = {};
    for (const key of Object.keys(outputs)) {
      const val = outputs[key];
      result[key] = val && val.shape && val.data ? wrapTensor(val) : val;
    }
    return result;
  }

  /**
   * Executes the pipeline asynchronously on a worker thread, returning a Promise.
   * @param {Float32Array | Uint8Array | Buffer | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string }} [input]
   * @returns {Promise<any>}
   */
  async run(input) {
    const res = await this._native.run(input);
    if (res && res.shape && res.data) {
      return wrapTensor(res);
    }
    return res;
  }

  /**
   * Executes the pipeline asynchronously returning a Promise with all named streams.
   * @param {Float32Array | Uint8Array | Buffer | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string }} [input]
   * @returns {Promise<Record<string, any>>}
   */
  async runAll(input) {
    const outputs = await this._native.runAll(input);
    const result = {};
    for (const key of Object.keys(outputs)) {
      const val = outputs[key];
      result[key] = val && val.shape && val.data ? wrapTensor(val) : val;
    }
    return result;
  }

  /**
   * Executes the pipeline synchronously with positional parameters bound in
   * declaration order. Each value is a tensor/audio/image object, a typed
   * array, a Buffer, a plain number (Scalar or *Arg), a string (StrArg), or a
   * boolean (BoolArg).
   * @param {...(Float32Array | Uint8Array | Buffer | number | string | boolean | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {any}
   */
  runSyncArgs(...args) {
    const res = this._native.runSyncArgs(args);
    if (res && res.shape && res.data) {
      return wrapTensor(res);
    }
    return res;
  }

  /**
   * Executes the pipeline synchronously with positional parameters, returning
   * a dictionary of all named streams.
   * @param {...(Float32Array | Uint8Array | Buffer | number | string | boolean | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {Record<string, any>}
   */
  runSyncAllArgs(...args) {
    const outputs = this._native.runSyncAllArgs(args);
    const result = {};
    for (const key of Object.keys(outputs)) {
      const val = outputs[key];
      result[key] = val && val.shape && val.data ? wrapTensor(val) : val;
    }
    return result;
  }

  /**
   * Executes the pipeline asynchronously with positional parameters, returning
   * a Promise.
   * @param {...(Float32Array | Uint8Array | Buffer | number | string | boolean | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {Promise<any>}
   */
  async runArgs(...args) {
    const res = await this._native.runArgs(args);
    if (res && res.shape && res.data) {
      return wrapTensor(res);
    }
    return res;
  }

  /**
   * Executes the pipeline asynchronously with positional parameters, returning
   * a Promise with all named streams.
   * @param {...(Float32Array | Uint8Array | Buffer | number | string | boolean | { data: Buffer | ArrayBuffer, shape: number[], dtype?: string, payloadType?: string })} args
   * @returns {Promise<Record<string, any>>}
   */
  async runAllArgs(...args) {
    const outputs = await this._native.runAllArgs(args);
    const result = {};
    for (const key of Object.keys(outputs)) {
      const val = outputs[key];
      result[key] = val && val.shape && val.data ? wrapTensor(val) : val;
    }
    return result;
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
