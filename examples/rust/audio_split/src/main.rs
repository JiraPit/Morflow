use core_types::Payload;
use morflow::Morflow;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    // 1. Load pipeline and read input WAV binary
    let mut pipeline = Morflow::load("audio_split_pipeline.morf")?;
    let input_bytes = fs::read("input.wav")?;

    println!("Loaded pipeline: audio_split_pipeline.morf");
    println!("Input audio binary: {} bytes", input_bytes.len());

    // 2. Execute pipeline with input binary payload
    let outputs = pipeline.run(Payload::Data {
        buffer: core_types::RVec::from(input_bytes),
    })?;

    // 3. Save each emitted channel directly to its own WAV file
    for (name, payload) in outputs.into_iter() {
        if let Payload::Data { buffer } = payload {
            let filename = format!("{}.wav", name);
            fs::write(&filename, buffer.as_slice())?;
            println!(
                "Saved channel output '{}' ({} bytes) to {}",
                name,
                buffer.len(),
                filename
            );
        }
    }

    println!("Multi-channel audio split completed successfully!");
    Ok(())
}
