use core_types::{Audio, Payload};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use morflow::Morflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    // 1. Load pipeline and decode input WAV
    let mut pipeline = Morflow::load("audio_pipeline.morf")?;
    let mut reader = WavReader::open("input.wav")?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let mut planar = vec![Vec::new(); channels];
    for (i, sample) in reader.samples::<i16>().enumerate() {
        planar[i % channels].push(sample? as f32 / 32768.0);
    }
    let samples: Vec<f32> = planar.into_iter().flatten().collect();

    // 2. Wrap into Morflow Audio and execute DSP pipeline
    let input =
        Audio::from_f32_planar(&samples, channels, spec.sample_rate).map_err(|e| e.to_string())?;
    let outputs = pipeline.run(Payload::Audio(input))?;

    // 3. Extract emitted output and save output WAV
    let out_samples = match outputs.into_single()? {
        Payload::Audio(out) => out.to_vec_f32(),
        Payload::Tensor(out) => out.to_vec_f32(),
        _ => return Err("Unexpected output payload".into()),
    };

    let out_spec = WavSpec {
        channels: channels as u16,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create("output.wav", out_spec)?;
    let num_samples = out_samples.len() / channels;
    for s in 0..num_samples {
        for c in 0..channels {
            writer.write_sample(out_samples[c * num_samples + s])?;
        }
    }
    writer.finalize()?;
    println!("Successfully processed audio and saved to output.wav");

    Ok(())
}
