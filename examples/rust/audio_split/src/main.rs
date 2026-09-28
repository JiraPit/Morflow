use core_types::{Audio, Payload};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use morflow::Morflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    // 1. Load pipeline and decode input WAV
    let mut pipeline = Morflow::load("audio_split_pipeline.morf")?;
    let mut reader = WavReader::open("input.wav")?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let mut planar = vec![Vec::new(); channels];
    for (i, sample) in reader.samples::<i16>().enumerate() {
        planar[i % channels].push(sample? as f32 / 32768.0);
    }
    let samples: Vec<f32> = planar.into_iter().flatten().collect();

    // 2. Wrap into Morflow Audio and execute pipeline
    let input =
        Audio::from_f32_planar(&samples, channels, spec.sample_rate).map_err(|e| e.to_string())?;
    let outputs = pipeline.run(Payload::Audio(input))?;

    // 3. Save each emitted channel to its own WAV file
    for (name, payload) in outputs.iter() {
        let out_spec = WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(format!("{}.wav", name), out_spec)?;
        if let Payload::Audio(aud) = payload {
            for sample in aud.to_vec_f32() {
                writer.write_sample(sample)?;
            }
        }
        writer.finalize()?;
        println!("Saved channel output to {}.wav", name);
    }

    Ok(())
}
