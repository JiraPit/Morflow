import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../../../bindings/js/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function decodeWav(buffer) {
  const numChannels = buffer.readUInt16LE(22);
  const sampleRate = buffer.readUInt32LE(24);
  const bitsPerSample = buffer.readUInt16LE(34);
  
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

  // 1. Load pipeline and decode stereo WAV
  const pipeline = morflow.load('audio_split_pipeline.morf');
  const inputBuffer = fs.readFileSync('input.wav');
  const wav = decodeWav(inputBuffer);

  console.log(`Loaded pipeline: ${pipeline.name || 'audio_split_pipeline'}`);
  console.log(`Input audio: ${wav.numChannels} channels, ${wav.sampleRate} Hz, ${wav.numSamples} samples`);

  const flattened = new Float32Array(wav.numChannels * wav.numSamples);
  for (let ch = 0; ch < wav.numChannels; ch++) {
    flattened.set(wav.planar[ch], ch * wav.numSamples);
  }

  const tensorInput = {
    data: Buffer.from(flattened.buffer),
    shape: [wav.numChannels, wav.numSamples],
    dtype: 'f32'
  };

  // 2. Execute multi-output Morflow pipeline
  const outputs = await pipeline.runAll(tensorInput);

  // 3. Save separated channels
  if (outputs.left_filtered) {
    const leftData = outputs.left_filtered.toFloat32Array();
    const encodedLeft = encodeWav([leftData], 44100);
    fs.writeFileSync('left_filtered.wav', encodedLeft);
    console.log('Saved left channel to left_filtered.wav');
  }

  if (outputs.right_filtered) {
    const rightData = outputs.right_filtered.toFloat32Array();
    const encodedRight = encodeWav([rightData], 44100);
    fs.writeFileSync('right_filtered.wav', encodedRight);
    console.log('Saved right channel to right_filtered.wav');
  }

  console.log('Successfully processed multi-output audio split in JavaScript!');
}

main().catch(console.error);
