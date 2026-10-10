import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('Morflow - fromStr compilation and metadata', () => {
  const dsl = `
    accept Audio $audio_in
    accept IntArg $rate = 44100
    
    $audio_in >> base/latest/identity >> emit
  `;
  const pipeline = morflow.fromStr(dsl);
  assert.equal(pipeline.params.length, 2);
  assert.equal(pipeline.params[0], 'audio_in');
  assert.equal(pipeline.params[1], 'rate');
});

test('Morflow - synchronous execution with Float32Array', () => {
  const dsl = `
    accept Tensor $tensor
    $tensor >> base/latest/identity >> emit
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
    accept Tensor $tensor
    $tensor >> base/latest/identity >> emit
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
    accept Tensor $img
    $img[1:3, :] >> base/latest/identity >> emit
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
    accept Tensor $audio
    $audio[0:2] >> base/latest/identity >> emit("low")
    $audio[2:4] >> base/latest/identity >> emit("high")
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
    morflow.fromStr('accept Tensor $x >>> broken');
  }, /Parse error/);
});

test('Morflow - audio to_audio and to_wav pipeline', async () => {
  const dsl = `
    import audio_basics/latest
    accept Bytes $data
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

test('Morflow - bare rank-2 array is a plain tensor', () => {
  // There is no type inference: a bare [4, N] feature matrix is a tensor and
  // reaches tensor actions without any wrapping.
  const pipeline = morflow.fromStr(`
    import tensor_basics/latest
    accept Tensor $data
    $data >> reshape(shape="2, 1000") >> emit
  `);

  const data = Buffer.alloc(2000 * 4);
  for (let i = 0; i < 2000; i++) data.writeFloatLE(i, i * 4);

  const output = pipeline.runSync({ data, shape: [4, 500], dtype: 'f32' });
  assert.deepEqual(output.shape, [2, 1000]);
});

test('Morflow - payloadType tensor forces a plain tensor', () => {
  const pipeline = morflow.fromStr(`
    import tensor_basics/latest
    accept Tensor $data
    $data >> reshape(shape="2, 1000") >> emit
  `);

  const data = Buffer.alloc(2000 * 4);
  for (let i = 0; i < 2000; i++) data.writeFloatLE(i, i * 4);

  const output = pipeline.runSync({ data, shape: [4, 500], dtype: 'f32', payloadType: 'tensor' });
  assert.deepEqual(output.shape, [2, 1000]);
  assert.equal(output.toFloat32Array()[0], 0);
  assert.equal(output.toFloat32Array()[1999], 1999);
});

test('Morflow - payloadType audio forces an audio payload', async () => {
  const pipeline = morflow.fromStr(`
    import audio_basics/latest
    accept Audio $audio
    $audio >> to_wav >> emit
  `);

  const data = Buffer.alloc(2 * 1000 * 4);
  const output = await pipeline.run({
    data,
    shape: [2, 1000],
    dtype: 'f32',
    payloadType: 'audio',
    sampleRate: 48000
  });

  const wav = output.toBuffer();
  assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
  assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
});

test('Morflow - payloadType image forces an image payload', () => {
  const pipeline = morflow.fromStr(`
    from base/latest import to_tensor
    from image_basics/latest import to_image
    accept Image $image
    $image >> to_tensor >> to_image >> emit
  `);

  const output = pipeline.runSync({
    data: Buffer.alloc(8 * 8 * 4),
    shape: [8, 8, 1],
    dtype: 'f32',
    payloadType: 'image',
    colorSpace: 'grayscale'
  });
  assert.deepEqual(output.shape, [8, 8]);
});

test('Morflow - payloadType rejects unknown values', () => {
  const pipeline = morflow.fromStr(`
    accept Tensor $data
    $data >> base/latest/identity >> emit
  `);
  const data = Buffer.alloc(8 * 8 * 4);
  assert.throws(() => {
    pipeline.runSync({ data, shape: [8, 8], dtype: 'f32', payloadType: 'bogus' });
  }, /Unknown payloadType/);
});

test('Morflow - colorSpace rejects unknown values', () => {
  const pipeline = morflow.fromStr(`
    accept Tensor $data
    $data >> base/latest/identity >> emit
  `);
  const data = Buffer.alloc(8 * 8 * 3 * 4);
  assert.throws(() => {
    pipeline.runSync({ data, shape: [8, 8, 3], dtype: 'f32', payloadType: 'image', colorSpace: 'cmyk' });
  }, /Unknown color space/);
});

test('Morflow - bare rank-2 array is not inferred as audio', () => {
  // A bare [2, N] array is a plain tensor, so an audio-native action rejects
  // it. Set payloadType: 'audio' to send an audio payload.
  const pipeline = morflow.fromStr(`
    from audio_basics/latest import to_wav
    accept Audio $audio
    $audio >> to_wav >> emit
  `);
  const data = Buffer.alloc(2 * 1000 * 4);
  assert.throws(() => {
    pipeline.runSync({ data, shape: [2, 1000], dtype: 'f32' });
  }, /expected Audio/);

  const output = pipeline.runSync({
    data, shape: [2, 1000], dtype: 'f32', payloadType: 'audio', sampleRate: 44100
  });
  assert.equal(output.toBuffer().subarray(0, 4).toString(), 'RIFF');
});

test('Morflow - bare rank-3 array runs as a plain tensor', () => {
  // A rank-3 array is no longer auto-promoted to an image. Image actions
  // declare DataType::Tensor input, so they still accept it directly.
  const pipeline = morflow.fromStr(`
    from image_basics/latest import to_image
    accept Tensor $image
    $image >> to_image >> emit
  `);
  const output = pipeline.runSync({ data: Buffer.alloc(8 * 8 * 3), shape: [8, 8, 3], dtype: 'u8' });
  assert.deepEqual(output.shape, [8, 8, 3]);
  assert.equal(output.dtype, 'u8');
});

test('Morflow - scalar param accepts a plain number', () => {
  const pipeline = morflow.fromStr(`
    import nn_basics/latest
    accept Scalar $value
    $value >> relu >> emit
  `);
  const output = pipeline.runSyncArgs(-3.0);
  assert.ok(output);
  assert.deepEqual(Array.from(output.toFloat32Array()), [0.0]);
});

test('Morflow - IntArg positional parameter with default', async () => {
  const pipeline = morflow.fromStr(`
    import audio_basics/latest
    accept Bytes $data
    accept IntArg $rate = 48000
    $data >> to_audio(channels=2, sample_rate=$rate, dtype="i16") >> to_wav >> emit
  `);
  const pcm = Buffer.alloc(8);

  const viaDefault = await pipeline.runArgs(pcm);
  assert.equal(viaDefault.toBuffer().toString('ascii', 0, 4), 'RIFF');

  const viaArg = await pipeline.runArgs(pcm, 22050);
  assert.equal(viaArg.toBuffer().toString('ascii', 0, 4), 'RIFF');
});

test('Morflow - StrArg, BoolArg and Scalar positional parameters', () => {
  const pipeline = morflow.fromStr(`
    import base/latest
    from audio_basics/latest import to_wav
    accept Audio $audio
    accept FloatArg $linear = 1.0
    accept BoolArg $routed = true
    accept Scalar $mix = 0.0
    $audio >> to_wav >> emit
  `);
  const data = Buffer.alloc(2 * 64 * 4);
  const outputs = pipeline.runSyncAllArgs(
    { data, shape: [2, 64], dtype: 'f32', payloadType: 'audio', sampleRate: 44100 },
    2.0,
    false,
    0.5
  );
  assert.ok(outputs);
});

test('Morflow - exact version and required imports', () => {
  const pipeline = morflow.fromStr(`from base/0.3.2 import identity
accept Tensor $data
$data >> identity >> emit`);
  assert.deepEqual(Array.from(pipeline.runSync(new Float32Array([1, 2])).toFloat32Array()), [1, 2]);
  assert.throws(() => morflow.fromStr(`accept Tensor $data
$data >> identity >> emit`), /not declared by the imports/);
});

test('Morflow - select QR Composite components', () => {
  const pipeline = morflow.fromStr(`
    import linalg_basics/latest
    accept Tensor[3,2] $matrix
    $matrix >> qr >> $parts
    $parts[0] >> emit("q")
    $parts[1] >> emit("r")
        $parts >> matmul >> emit("reconstructed")
  `);
  const data = new Float32Array([1, 0, 0, 1, 1, 1]);
  const out = pipeline.runSyncAll({data: Buffer.from(data.buffer), shape: [3, 2], dtype: 'f32'});
  assert.deepEqual(out.q.shape, [3, 2]);
  assert.deepEqual(out.r.shape, [2, 2]);
  assert.deepEqual(out.reconstructed.shape, [3, 2]);
  const reconstructed = out.reconstructed.toFloat32Array();
  for (let i = 0; i < data.length; i++) assert.ok(Math.abs(reconstructed[i] - data[i]) < 1e-5);
  const q = out.q.toFloat32Array();
  const r = out.r.toFloat32Array();
  for (let i = 0; i < 6; i++) {
    const row = Math.floor(i / 2), col = i % 2;
    assert.ok(Math.abs(q[row * 2] * r[col] + q[row * 2 + 1] * r[2 + col] - data[i]) < 1e-5);
  }
});


test('Morflow - declared plugin works through file and string APIs', {skip: !process.env.MORFLOW_PLUGINS_PATH}, async () => {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'morflow-plugin-'));
  const source = `plugin opencv-bridge/0.1.2
from image_opencv/0.1.2 import resize
accept Tensor[6,8,3] $image
$image >> resize(4,3,filter="nearest") >> emit`;
  const file = path.join(directory, 'plugin.morf');
  fs.writeFileSync(file, source);
  try {
    const values = new Float32Array(6 * 8 * 3).fill(0.5);
    const input = {data: Buffer.from(values.buffer), shape: [6,8,3], dtype: 'f32'};
    for (const pipeline of [morflow.load(file), morflow.fromStr(source)]) {
      for (const output of [pipeline.runSync(input), ...await Promise.all([pipeline.run(input), pipeline.run(input)])]) {
        assert.deepEqual(output.shape, [3,4,3]);
        assert.ok(Array.from(output.toFloat32Array()).every(value => value === 0.5));
      }
    }
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});
