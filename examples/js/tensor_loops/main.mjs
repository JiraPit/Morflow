import path from 'node:path';
import { fileURLToPath } from 'node:url';
import morflow from '../../../bindings/js/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  process.chdir(__dirname);

  // 1. Load pipeline and build a [4, 3] tensor: one row per time step, one column per sensor
  const pipeline = morflow.load('tensor_loops_pipeline.morf');
  const readings = new Float32Array([
    -1.0, 2.0, 0.5,
    3.0, -1.0, 1.5,
    2.0, 4.0, -0.5,
    0.5, 1.0, 12.0,
  ]);

  console.log('Loaded pipeline: tensor_loops_pipeline.morf');
  console.log('Input readings: shape [4, 3]');

  // 2. Execute Morflow pipeline asynchronously
  const outputs = await pipeline.runToMap({ data: readings, shape: [4, 3], dtype: 'f32' });

  // 3. Print each emitted tensor, one row per line
  for (const name of ['running_totals', 'step_weights', 'aligned', 'checked']) {
    const tensor = outputs[name];
    const values = tensor.toFloat32Array();
    const columns = tensor.shape[1];
    console.log(`${name} [${tensor.shape.join(', ')}]:`);
    for (let row = 0; row < tensor.shape[0]; row++) {
      console.log(`  [${Array.from(values.subarray(row * columns, (row + 1) * columns)).join(', ')}]`);
    }
  }
}

main().catch(console.error);
