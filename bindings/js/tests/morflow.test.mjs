import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('Morflow - fromStr compilation and metadata', () => {
  const dsl = `
    accept $audio_in
    accept $rate = 44100
    
    $audio_in >> identity >> emit
  `;
  const pipeline = morflow.fromStr(dsl);
  assert.equal(pipeline.params.length, 2);
  assert.equal(pipeline.params[0], 'audio_in');
  assert.equal(pipeline.params[1], 'rate');
});

test('Morflow - synchronous execution with Float32Array', () => {
  const dsl = `
    accept $tensor
    $tensor >> identity >> emit
  `;
  const pipeline = morflow.fromStr(dsl);
  const input = new Float32Array([1.0, 2.5, -3.0, 4.25]);

  const output = pipeline.runSync(input);
  assert.ok(output);
  assert.equal(output.dtype, 'f32');
  assert.deepEqual(output.shape, [4]);
  assert.equal(output.length, 4);

  const f32Out = output.toFloat32Array();
  assert.equal(f32Out.length, 4);
  assert.equal(f32Out[0], 1.0);
  assert.equal(f32Out[1], 2.5);
  assert.equal(f32Out[2], -3.0);
  assert.equal(f32Out[3], 4.25);
});

test('Morflow - asynchronous execution with Promise', async () => {
  const dsl = `
    accept $tensor
    $tensor >> identity >> emit
  `;
  const pipeline = morflow.fromStr(dsl);
  const input = new Float32Array([10.0, 20.0, 30.0]);

  const output = await pipeline.run(input);
  assert.ok(output);
  assert.equal(output.dtype, 'f32');
  assert.deepEqual(output.shape, [3]);

  const f32Out = output.toFloat32Array();
  assert.deepEqual(Array.from(f32Out), [10.0, 20.0, 30.0]);
});

test('Morflow - multi-dimensional TensorInput', () => {
  const dsl = `
    accept $img
    $img[1:3, :] >> identity >> emit
  `;
  const pipeline = morflow.fromStr(dsl);

  // 4x4 F32 matrix
  const matrix = new Float32Array(16);
  for (let i = 0; i < 16; i++) {
    matrix[i] = i * 1.0;
  }

  const tensorInput = {
    data: Buffer.from(matrix.buffer),
    shape: [4, 4],
    dtype: 'f32'
  };

  const output = pipeline.runSync(tensorInput);
  assert.ok(output);
  assert.deepEqual(output.shape, [2, 4]);
  assert.equal(output.length, 8);

  const f32Out = output.toFloat32Array();
  assert.equal(f32Out[0], 4.0); // row 1, col 0
  assert.equal(f32Out[7], 11.0); // row 2, col 3
});

test('Morflow - multiple named outputs (runSyncAll & runAll)', async () => {
  const dsl = `
    accept $audio
    $audio[0:2] >> identity >> emit("low")
    $audio[2:4] >> identity >> emit("high")
  `;
  const pipeline = morflow.fromStr(dsl);
  const input = new Float32Array([1.0, 2.0, 3.0, 4.0]);

  // Sync test
  const syncOut = pipeline.runSyncAll(input);
  assert.ok(syncOut.low);
  assert.ok(syncOut.high);
  assert.deepEqual(Array.from(syncOut.low.toFloat32Array()), [1.0, 2.0]);
  assert.deepEqual(Array.from(syncOut.high.toFloat32Array()), [3.0, 4.0]);

  // Async test
  const asyncOut = await pipeline.runAll(input);
  assert.ok(asyncOut.low);
  assert.ok(asyncOut.high);
  assert.deepEqual(Array.from(asyncOut.low.toFloat32Array()), [1.0, 2.0]);
  assert.deepEqual(Array.from(asyncOut.high.toFloat32Array()), [3.0, 4.0]);
});

test('Morflow - error handling on invalid pipeline syntax', () => {
  assert.throws(() => {
    morflow.fromStr('invalid syntax >> >> >>>');
  }, /Parse error/);
});

test('Morflow - audio to_audio and to_wav pipeline', async () => {
  const dsl = `
    import audio_essentials.latest
    accept $data
    $data >> to_audio(channels=2, sample_rate=44100, dtype="i16") >> gain(linear=2.0) >> to_wav >> emit
  `;
  const pipeline = morflow.fromStr(dsl);
  const pcmBuffer = Buffer.alloc(8);
  pcmBuffer.writeInt16LE(16384, 0);
  pcmBuffer.writeInt16LE(-8192, 2);
  pcmBuffer.writeInt16LE(16384, 4);
  pcmBuffer.writeInt16LE(-8192, 6);

  const output = await pipeline.run(pcmBuffer);
  const wavBuf = output.toBuffer();
  assert.equal(wavBuf.length, 44 + 8);
  assert.equal(wavBuf.toString('ascii', 0, 4), 'RIFF');
  assert.equal(wavBuf.toString('ascii', 8, 12), 'WAVE');
});
