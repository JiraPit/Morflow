use core_types::{DataType, Payload, RVec, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::RawBytes
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut target_dtype = "i16";
    let mut target_sample_rate: Option<u32> = None;
    let mut clip = true;

    if let Some(args) = payload.args() {
        if let Some(dt) = args
            .get_named("dtype")
            .or_else(|| args.get_named("format"))
            .or_else(|| args.get_named("bits"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            target_dtype = dt;
        }
        if let Some(sr_str) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
        {
            if let Ok(sr) = sr_str.parse::<u32>() {
                target_sample_rate = Some(sr);
            }
        }
        if let Some(clip_str) = args.get_named("clip") {
            clip = !matches!(
                clip_str.to_lowercase().as_str(),
                "false" | "0" | "no" | "off"
            );
        }
    }

    let (planar_f32, channels, num_samples, sample_rate) = match payload.unwrap_payload() {
        Payload::Audio(audio) => {
            let ch = audio.channels().max(1);
            let samples = audio.num_samples();
            let sr = target_sample_rate.unwrap_or(audio.sample_rate);
            let f32_vec = audio.to_vec_f32();
            (f32_vec, ch, samples, sr)
        }
        Payload::Tensor(tensor) => {
            let shape = tensor.shape.as_slice();
            let (ch, samples) = match shape.len() {
                1 => (1, shape[0]),
                2 => (shape[0], shape[1]),
                _ => (1, tensor.num_elements()),
            };
            let sr = target_sample_rate.unwrap_or(44100);
            let f32_vec = if tensor.dtype == TensorDType::F32 {
                tensor.to_vec_f32()
            } else {
                let bytes = tensor.to_contiguous_bytes();
                bytes.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
            };
            (f32_vec, ch, samples, sr)
        }
        Payload::Data { buffer } => {
            // If already a WAV file buffer, pass through
            if buffer.len() >= 12 && &buffer[0..4] == b"RIFF" && &buffer[8..12] == b"WAVE" {
                return Payload::Data {
                    buffer: buffer.clone(),
                };
            }
            return Payload::Data {
                buffer: buffer.clone(),
            };
        }
        other => return other.clone(),
    };

    let wav_bytes = encode_wav_binary(
        &planar_f32,
        channels,
        num_samples,
        sample_rate,
        target_dtype,
        clip,
    );

    Payload::Data {
        buffer: RVec::from(wav_bytes),
    }
}

/// Encodes planar Float32 audio samples into a complete RIFF/WAVE file buffer.
pub fn encode_wav_binary(
    planar_f32: &[f32],
    channels: usize,
    num_samples: usize,
    sample_rate: u32,
    dtype: &str,
    clip: bool,
) -> Vec<u8> {
    let clamp_sample = |s: f32| -> f32 {
        if clip {
            s.clamp(-1.0, 1.0)
        } else {
            s
        }
    };

    let (audio_format, bits_per_sample, bytes_per_sample): (u16, u16, usize) =
        match dtype.to_lowercase().as_str() {
            "f32" | "float" | "float32" | "32f" => (3, 32, 4),
            "i24" | "int24" | "24" => (1, 24, 3),
            "i32" | "int32" | "32" => (1, 32, 4),
            "u8" | "uint8" | "8" => (1, 8, 1),
            _ /* "i16" / "int16" / "16" */ => (1, 16, 2),
        };

    let data_size = channels * num_samples * bytes_per_sample;
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
    let block_align = (channels * bytes_per_sample) as u16;

    let mut wav = Vec::with_capacity(44 + data_size);

    // 1. RIFF Header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((36 + data_size) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // 2. fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size = 16
    wav.extend_from_slice(&audio_format.to_le_bytes());
    wav.extend_from_slice(&(channels as u16).to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // 3. data chunk header
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_size as u32).to_le_bytes());

    // 4. Interleaved sample data
    let mut data_buf = vec![0u8; data_size];

    match bits_per_sample {
        16 => {
            data_buf
                .par_chunks_exact_mut(channels * 2)
                .enumerate()
                .for_each(|(s, frame)| {
                    for c in 0..channels {
                        let val = clamp_sample(planar_f32[c * num_samples + s]);
                        let i16_val = (val * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
                        frame[c * 2..c * 2 + 2].copy_from_slice(&i16_val.to_le_bytes());
                    }
                });
        }
        24 => {
            data_buf
                .par_chunks_exact_mut(channels * 3)
                .enumerate()
                .for_each(|(s, frame)| {
                    for c in 0..channels {
                        let val = clamp_sample(planar_f32[c * num_samples + s]);
                        let i24 = (val * 8388607.0).round().clamp(-8388608.0, 8388607.0) as i32;
                        let b0 = (i24 & 0xFF) as u8;
                        let b1 = ((i24 >> 8) & 0xFF) as u8;
                        let b2 = ((i24 >> 16) & 0xFF) as u8;
                        frame[c * 3] = b0;
                        frame[c * 3 + 1] = b1;
                        frame[c * 3 + 2] = b2;
                    }
                });
        }
        32 if audio_format == 3 => {
            // 32-bit IEEE Float
            data_buf
                .par_chunks_exact_mut(channels * 4)
                .enumerate()
                .for_each(|(s, frame)| {
                    for c in 0..channels {
                        let val = clamp_sample(planar_f32[c * num_samples + s]);
                        frame[c * 4..c * 4 + 4].copy_from_slice(&val.to_le_bytes());
                    }
                });
        }
        32 => {
            // 32-bit integer PCM
            data_buf
                .par_chunks_exact_mut(channels * 4)
                .enumerate()
                .for_each(|(s, frame)| {
                    for c in 0..channels {
                        let val = clamp_sample(planar_f32[c * num_samples + s]);
                        let i32_val = (val * 2147483647.0).round().clamp(-2147483648.0, 2147483647.0) as i32;
                        frame[c * 4..c * 4 + 4].copy_from_slice(&i32_val.to_le_bytes());
                    }
                });
        }
        8 => {
            // 8-bit unsigned PCM
            data_buf
                .par_chunks_exact_mut(channels)
                .enumerate()
                .for_each(|(s, frame)| {
                    for c in 0..channels {
                        let val = clamp_sample(planar_f32[c * num_samples + s]);
                        let u8_val = ((val * 127.0).round().clamp(-128.0, 127.0) + 128.0) as u8;
                        frame[c] = u8_val;
                    }
                });
        }
        _ => {}
    }

    wav.extend_from_slice(&data_buf);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_to_wav_16bit_stereo() {
        // 2 channels, 100 samples each
        let planar = vec![0.5f32; 200];
        let audio = Audio::from_f32_planar(&planar, 2, 48000).unwrap();

        let payload = Payload::Audio(audio);
        let result = process(payload);

        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.len(), 44 + 200 * 2);
            assert_eq!(&buffer[0..4], b"RIFF");
            assert_eq!(&buffer[8..12], b"WAVE");
            assert_eq!(&buffer[12..16], b"fmt ");
            assert_eq!(&buffer[36..40], b"data");

            let channels = u16::from_le_bytes(buffer[22..24].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(buffer[24..28].try_into().unwrap());
            let bits = u16::from_le_bytes(buffer[34..36].try_into().unwrap());

            assert_eq!(channels, 2);
            assert_eq!(sample_rate, 48000);
            assert_eq!(bits, 16);

            // First sample left & right
            let s0_left = i16::from_le_bytes(buffer[44..46].try_into().unwrap());
            let s0_right = i16::from_le_bytes(buffer[46..48].try_into().unwrap());
            assert_eq!(s0_left, 16384);
            assert_eq!(s0_right, 16384);
        } else {
            panic!("Expected Payload::Data");
        }
    }

    #[test]
    fn test_to_wav_f32_float() {
        let planar = vec![0.75f32, -0.75f32];
        let audio = Audio::from_f32_planar(&planar, 1, 96000).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dtype"), RString::from("f32")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Audio(audio)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.len(), 44 + 2 * 4);
            let audio_format = u16::from_le_bytes(buffer[20..22].try_into().unwrap());
            let bits = u16::from_le_bytes(buffer[34..36].try_into().unwrap());
            assert_eq!(audio_format, 3); // IEEE Float
            assert_eq!(bits, 32);

            let f0 = f32::from_le_bytes(buffer[44..48].try_into().unwrap());
            let f1 = f32::from_le_bytes(buffer[48..52].try_into().unwrap());
            assert_eq!(f0, 0.75);
            assert_eq!(f1, -0.75);
        } else {
            panic!("Expected Payload::Data");
        }
    }
}
