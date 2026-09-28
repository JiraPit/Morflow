const path = require('path');
const fs = require('fs');

let nativeBinding = null;

// Search for the platform-specific compiled .node binary
const possiblePaths = [
  path.join(__dirname, 'morflow.linux-x64-gnu.node'),
  path.join(__dirname, 'morflow.node'),
  path.join(__dirname, '..', '..', 'target', 'release', 'libmorflow_node.so'),
  path.join(__dirname, '..', '..', 'target', 'debug', 'libmorflow_node.so')
];

for (const candidate of possiblePaths) {
  if (fs.existsSync(candidate)) {
    try {
      nativeBinding = require(candidate);
      break;
    } catch (e) {
      // try next candidate
    }
  }
}

if (!nativeBinding) {
  // Fallback to direct require
  try {
    nativeBinding = require('./morflow.linux-x64-gnu.node');
  } catch (err) {
    throw new Error(`Failed to load Morflow native addon: ${err.message}`);
  }
}

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

  get name() {
    return this._native.name;
  }

  get params() {
    return this._native.params;
  }

  warmup() {
    return this._native.warmup();
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

module.exports = {
  load,
  fromStr,
  fromString: fromStr,
  wrapTensor,
  Pipeline: MorflowPipelineWrapper,
  MorflowTensor: nativeBinding.MorflowTensor
};
