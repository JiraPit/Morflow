use abi_stable::std_types::RVec;
use core_types::{Payload, Tensor};
use pipeline::Morflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Morflow Engine ===");

    // 1. Load pipeline directly from .morf definition with accept and resurface
    let mut pipeline = Morflow::from_str(
        r#"
        accept $input_data

        $input_data >> identity >> resurface
        "#,
    )?;

    // 2. Prepare payload
    let input_audio = Payload::Data {
        buffer: RVec::from(vec![10, 20, 30, 40]),
    };

    // 3. Run pipeline!
    println!("\nExecuting pipeline with input payload: {:?}", input_audio);
    let outputs = pipeline.run(input_audio)?;
    let processed = outputs.into_single()?;
    println!("Result from pipeline.run(): {:?}", processed);

    // 4. Run with Zero-Copy Tensor View
    let tensor_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let tensor = Tensor::from_f32_shape(&tensor_data, vec![2, 8])
        .map_err(|e| format!("Tensor error: {}", e))?;

    let tensor_payload = Payload::Tensor(tensor);
    println!("\nExecuting pipeline with Tensor payload...");
    let outputs = pipeline.run(tensor_payload)?;
    let processed_tensor = outputs.into_single()?;

    if let Payload::Tensor(out_t) = processed_tensor {
        println!(
            "Result Tensor: rank={}, shape={:?}, contiguous={}",
            out_t.rank(),
            out_t.shape,
            out_t.is_contiguous()
        );
    }

    Ok(())
}
