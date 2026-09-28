use core_types::Payload;
use morflow::Morflow;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    // 1. Load pipeline and read input WAV binary
    let mut pipeline = Morflow::load("audio_pipeline.morf")?;
    let input_bytes = fs::read("input.wav")?;

    println!("Loaded pipeline: audio_pipeline.morf");
    println!("Input audio binary: {} bytes", input_bytes.len());

    // 2. Wrap into Morflow Payload and execute DSP pipeline
    let outputs = pipeline.run(Payload::Data {
        buffer: core_types::RVec::from(input_bytes),
    })?;

    // 3. Extract emitted output and save directly to output.wav
    let out_payload = outputs.into_single()?;
    if let Payload::Data { buffer } = out_payload {
        fs::write("output.wav", buffer.as_slice())?;
        println!(
            "Output audio: {} bytes, saved directly to output.wav",
            buffer.len()
        );
        println!("Successfully processed audio with Rust host!");
    } else {
        return Err("Unexpected output payload (expected Payload::Data)".into());
    }

    Ok(())
}
