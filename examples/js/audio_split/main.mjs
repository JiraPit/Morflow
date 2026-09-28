import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../../../bindings/js/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  process.chdir(__dirname);

  // 1. Load pipeline and read input WAV buffer
  const pipeline = morflow.load('audio_split_pipeline.morf');
  const inputBuffer = fs.readFileSync('input.wav');

  console.log('Loaded pipeline: audio_split_pipeline.morf');
  console.log(`Input audio binary: ${inputBuffer.length} bytes`);

  // 2. Execute Morflow multi-channel split pipeline asynchronously
  const outputs = await pipeline.runAll(inputBuffer);
  console.log(`Emitted outputs: ${Object.keys(outputs).join(', ')}`);

  // 3. Save each emitted channel directly to its own WAV file
  for (const [name, tensor] of Object.entries(outputs)) {
    const outFilename = `${name}.wav`;
    const wavBuffer = tensor.toBuffer();
    fs.writeFileSync(outFilename, wavBuffer);
    console.log(`Saved channel output '${name}' (${wavBuffer.length} bytes) to ${outFilename}`);
  }

  console.log('Multi-channel audio split completed successfully!');
}

main().catch(console.error);
