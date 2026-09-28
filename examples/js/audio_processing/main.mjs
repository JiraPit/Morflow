import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../../../bindings/js/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper to decode 16-bit PCM WAV file into planar Float32Array [channels, samples]
function decodeWav(buffer) {
  const numChannels = buffer.readUInt16LE(22);
  const sampleRate = buffer.readUInt32LE(24);
  const bitsPerSample = buffer.readUInt16LE(34);
  
  // Find data chunk
  let offset = 12;
  while (offset < buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    if (chunkId === 'data') {
      const dataOffset = offset + 8;
      const numSamples = chunkSize / (bitsPerSample / 8) / numChannels;
      const planar = [];
      for (let ch = 0; ch < numChannels; ch++) {
        planar.push(new Float32Array(numSamples));
      }
      for (let i = 0; i < numSamples; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
          const sampleIdx = dataOffset + (i * numChannels + ch) * 2;
          const int16 = buffer.readInt16LE(sampleIdx);
          planar[ch][i] = int16 / 32768.0;
        }
      }
      return { numChannels, sampleRate, numSamples, planar };
    }
    offset += 8 + chunkSize;
  }
  throw new Error('No data chunk found in WAV');
}

// Helper to encode planar Float32Array into 16-bit PCM WAV
function encodeWav(planar, sampleRate) {
  const numChannels = planar.length;
  const numSamples = planar[0].length;
  const byteRate = sampleRate * numChannels * 2;
  const blockAlign = numChannels * 2;
  const dataSize = numSamples * numChannels * 2;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(16, 34); // 16 bits
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const val = Math.max(-1.0, Math.min(1.0, planar[ch][i]));
      const int16 = Math.round(val * 32767.0);
      buffer.writeInt16LE(int16, offset);
      offset += 2;
    }
  }
  return buffer;
}

async function main() {
  process.chdir(__dirname);

  // 1. Load pipeline and decode input WAV
  const pipeline = morflow.load('audio_pipeline.morf');
  const inputBuffer = fs.readFileSync('input.wav');
  const wav = decodeWav(inputBuffer);

  console.log('Loaded pipeline: audio_pipeline.morf');
  console.log(`Input audio: ${wav.numChannels} channels, ${wav.sampleRate} Hz, ${wav.numSamples} samples`);

  // Flatten planar data for Morflow input: shape [channels, samples]
  const flattened = new Float32Array(wav.numChannels * wav.numSamples);
  for (let ch = 0; ch < wav.numChannels; ch++) {
    flattened.set(wav.planar[ch], ch * wav.numSamples);
  }

  const tensorInput = {
    data: Buffer.from(flattened.buffer),
    shape: [wav.numChannels, wav.numSamples],
    dtype: 'f32'
  };

  // 2. Execute Morflow DSP pipeline asynchronously
  const outputTensor = await pipeline.run(tensorInput);
  const outShape = outputTensor.shape;
  const outChannels = outShape[0];
  const outSamples = outShape[1];
  const outData = outputTensor.toFloat32Array();

  const outPlanar = [];
  for (let ch = 0; ch < outChannels; ch++) {
    outPlanar.push(outData.subarray(ch * outSamples, (ch + 1) * outSamples));
  }

  // 3. Save output WAV (48000 Hz)
  const encoded = encodeWav(outPlanar, 48000);
  fs.writeFileSync('output.wav', encoded);
  console.log(`Output audio: ${outChannels} channels, 48000 Hz, saved to output.wav`);
  console.log('Successfully processed audio with JavaScript Node.js host!');
}

main().catch(console.error);
