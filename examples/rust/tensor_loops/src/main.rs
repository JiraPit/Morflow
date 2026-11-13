use core_types::{Payload, Tensor};
use morflow::Morflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    // 1. Load pipeline and build a [4, 3] tensor: one row per time step, one column per sensor
    let mut pipeline = Morflow::load("tensor_loops_pipeline.morf")?;
    let readings = [
        -1.0, 2.0, 0.5, 3.0, -1.0, 1.5, 2.0, 4.0, -0.5, 0.5, 1.0, 12.0f32,
    ];
    let input = Tensor::from_f32_shape(&readings, vec![4, 3]).map_err(|e| e.to_string())?;

    println!("Loaded pipeline: tensor_loops_pipeline.morf");
    println!("Input readings: shape [4, 3]");

    // 2. Execute Morflow pipeline
    let outputs = pipeline.run(Payload::Tensor(input))?;

    // 3. Print each emitted tensor, one row per line
    for name in ["running_totals", "step_weights", "aligned", "checked"] {
        if let Some(Payload::Tensor(tensor)) = outputs.get(name) {
            println!("{} {:?}:", name, tensor.shape.as_slice());
            for row in tensor.to_vec_f32().chunks(tensor.shape[1]) {
                println!("  {:?}", row);
            }
        }
    }

    Ok(())
}
