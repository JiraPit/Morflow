import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../../../bindings/js/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  process.chdir(__dirname);

  // 1. Load pipeline and read input WAV buffer
  const pipeline = morflow.load('audio_pipeline.morf');
  const inputBuffer = fs.readFileSync('input.wav');

  console.log('Loaded pipeline: audio_pipeline.morf');
  console.log(`Input audio binary: ${inputBuffer.length} bytes`);

  // 2. Execute Morflow DSP pipeline asynchronously (handles WAV decoding, DSP chain, and WAV encoding)
  const output = await pipeline.run(inputBuffer);
  const outputBuffer = output.toBuffer();

  // 3. Save output WAV directly to disk
  fs.writeFileSync('output.wav', outputBuffer);
  console.log(`Output audio: ${outputBuffer.length} bytes, saved to output.wav`);
  console.log('Successfully processed audio with JavaScript Node.js host!');
}

main().catch(console.error);
