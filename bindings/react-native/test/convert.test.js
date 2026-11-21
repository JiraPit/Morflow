'use strict';
// npm test (builds lib/ first)

const assert = require('node:assert/strict');
const test = require('node:test');

const { toNative, wrapOutput, wrapOutputs } = require('../lib/convert');

test('typed arrays become 1-D tensors of their dtype, and Uint8Arrays raw bytes', () => {
  const floats = new Float32Array([1, -2, 3]);
  const tensor = toNative(floats);
  assert.equal(tensor.kind, 'tensor');
  assert.equal(tensor.dtype, 'f32');
  assert.deepEqual(tensor.shape, [3]);
  assert.equal(tensor.data.byteLength, 12);
  assert.equal(tensor.data.buffer, floats.buffer, 'no copy');
  assert.equal(toNative(new BigInt64Array(2)).dtype, 'i64');
  const raw = toNative(new Uint8Array([1, 2]));
  assert.equal(raw.kind, 'bytes');
  assert.equal(toNative(new ArrayBuffer(4)).kind, 'bytes');
});

test('tensor objects keep their settings and infer the dtype from the data', () => {
  const samples = new Float64Array(8).subarray(2, 6);
  const input = toNative({ data: samples, shape: [1, 4], payloadType: 'audio', sampleRate: 16000 });
  assert.equal(input.dtype, 'f64');
  assert.equal(input.payloadType, 'audio');
  assert.equal(input.sampleRate, 16000);
  assert.equal(input.data.byteOffset, 16);
  assert.equal(input.data.byteLength, 32);
  assert.equal(toNative({ data: new Uint8Array(4), shape: [4] }).dtype, undefined, "Uint8Array data keeps the 'f32' default");
  assert.throws(() => toNative({ data: new Uint8Array(4) }), /needs a shape/);
});

test('arrays become Composites, and scalars pass through', () => {
  const composite = toNative([new Float32Array(2), { data: new Int32Array(1), shape: [1] }]);
  assert.equal(composite.payloadType, 'composite');
  assert.deepEqual(composite.items.map((i) => i.dtype), ['f32', 'i32']);
  assert.throws(() => toNative([new Uint8Array(1)]), /Composite components/);
  assert.equal(toNative(2.5), 2.5);
  assert.equal(toNative('text'), 'text');
  assert.equal(toNative(true), true);
  assert.equal(toNative(undefined), undefined);
  assert.throws(() => toNative({}), /pipeline input must be/);
});

test('outputs become MorflowTensors with typed views', () => {
  const values = new Float32Array([1.5, 2, 3]);
  const tensor = wrapOutput({ dtype: 'f32', shape: [3], data: values.buffer.slice(0) });
  assert.equal(tensor.rank, 1);
  assert.equal(tensor.length, 3);
  assert.deepEqual([...tensor.toFloat32Array()], [1.5, 2, 3]);
  assert.deepEqual([...tensor.toTypedArray()], [1.5, 2, 3]);
  assert.equal(tensor.toBuffer().byteLength, 12);
  assert.ok(tensor.data instanceof Uint8Array);
});

test('unaligned data is copied before a typed view is made', () => {
  const tensor = wrapOutput({ dtype: 'f32', shape: [1], data: new Float32Array([7]).buffer });
  tensor.data = new Uint8Array([0, ...new Uint8Array(new Float32Array([7]).buffer)]).subarray(1);
  assert.deepEqual([...tensor.toFloat32Array()], [7]);
});

test('composite and named outputs', () => {
  const out = wrapOutput({
    dtype: 'composite',
    items: [
      { dtype: 'u8', shape: [2], data: new Uint8Array([1, 2]).buffer },
      { dtype: 'composite', items: [] },
    ],
  });
  assert.equal(out.length, 2);
  assert.deepEqual([...out[0].toUint8Array()], [1, 2]);
  assert.deepEqual(out[1], []);
  const named = wrapOutputs({ a: { dtype: 'i32', shape: [], data: new Int32Array([5]).buffer } });
  assert.deepEqual([...named.a.toInt32Array()], [5]);
  assert.equal(named.a.length, 1);
  assert.equal(wrapOutput(undefined), undefined);
});

test('the built typings keep the .morf module declaration', () => {
  // tsc drops a triple-slash reference unless it says preserve="true"; without
  // it, `import boost from './boost.morf'` fails to type-check in apps.
  const typings = require('node:fs').readFileSync(require.resolve('../lib/index.d.ts'), 'utf8');
  assert.match(typings, /^\/\/\/ <reference path="[^"]*morf\.d\.ts" preserve="true" \/>/m);
});
