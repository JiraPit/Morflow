#![allow(clippy::needless_range_loop)]

use core_types::{Audio, AudioChannelLayout, AudioLayout, DataType, Payload, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::RawBytes | DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut target_sample_rate: Option<u32> = None;
    let mut target_channels: Option<usize> = None;
    let mut target_dtype = "i16";
    let mut target_layout = "interleaved";
    let mut normalize = true;

    if let Some(args) = &args_opt {
        if let Some(sr_str) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(sr) = sr_str.parse::<u32>() {
                target_sample_rate = Some(sr);
            }
        }
        if let Some(ch_str) = args
            .get_named("channels")
            .or_else(|| args.get_named("channel_count"))
        {
            if let Ok(ch) = ch_str.parse::<usize>() {
                target_channels = Some(ch);
            }
        }
        if let Some(dt) = args.get_named("dtype").or_else(|| args.get_named("format")) {
            target_dtype = dt;
        }
        if let Some(lay) = args.get_named("layout") {
            target_layout = lay;
        }
        if let Some(norm_str) = args.get_named("normalize") {
            normalize = !matches!(
                norm_str.to_lowercase().as_str(),
                "false" | "0" | "no" | "off"
            );
        }
    }

    match inner_payload {
        Payload::Data { buffer } => {
            let bytes = buffer.as_slice();
            // 1. Check if buffer is a RIFF/WAVE file
            if let Some(wav_audio) = try_parse_wav(bytes, target_sample_rate) {
                return Payload::Audio(wav_audio);
            }

            // 2. Decode as raw PCM byte buffer
            let sr = target_sample_rate.unwrap_or(44100);
            let ch = target_channels.unwrap_or(2).max(1);
            let is_planar = target_layout.eq_ignore_ascii_case("planar")
                || target_layout.eq_ignore_ascii_case("non-interleaved");

            let planar_f32 =
                decode_raw_pcm_to_planar_f32(bytes, ch, target_dtype, is_planar, normalize);
            if planar_f32.is_empty() {
                let dummy = Tensor::from_f32_vec(Vec::new(), vec![ch, 0]).unwrap();
                return Payload::Audio(Audio {
                    tensor: dummy,
                    sample_rate: sr,
                    channel_layout: AudioChannelLayout::from_channel_count(ch),
                    layout: AudioLayout::Planar,
                });
            }

            let shape = if ch == 1 {
                vec![planar_f32.len()]
            } else {
                vec![ch, planar_f32.len() / ch]
            };
            match Tensor::from_f32_vec(planar_f32, shape) {
                Ok(tensor) => Payload::Audio(Audio {
                    tensor,
                    sample_rate: sr,
                    channel_layout: AudioChannelLayout::from_channel_count(ch),
                    layout: AudioLayout::Planar,
                }),
                Err(err) => Payload::Error(err),
            }
        }
        Payload::Tensor(tensor) => {
            let sr = target_sample_rate.unwrap_or(44100);
            let shape = tensor.shape.as_slice();

            let (channels, planar_f32) = match shape.len() {
                1 => {
                    let f32_vec = tensor_to_f32_vec(&tensor, normalize);
                    (1, f32_vec)
                }
                2 => {
                    let c = target_channels.unwrap_or(shape[0]);
                    let f32_vec = tensor_to_f32_vec(&tensor, normalize);
                    if c == shape[0] {
                        (c, f32_vec)
                    } else if shape[1] <= 8 && c == shape[1] {
                        // Tensor was [samples, channels], de-interleave to planar [channels, samples]
                        let num_samples = shape[0];
                        let num_channels = shape[1];
                        let mut planar = vec![0.0f32; num_channels * num_samples];
                        planar
                            .par_chunks_exact_mut(num_samples)
                            .enumerate()
                            .for_each(|(ch, plane)| {
                                for s in 0..num_samples {
                                    plane[s] = f32_vec[s * num_channels + ch];
                                }
                            });
                        (num_channels, planar)
                    } else {
                        (c, f32_vec)
                    }
                }
                _ => {
                    let f32_vec = tensor_to_f32_vec(&tensor, normalize);
                    (1, f32_vec)
                }
            };

            let shape = if channels == 1 {
                vec![planar_f32.len()]
            } else {
                vec![channels, planar_f32.len() / channels]
            };
            match Tensor::from_f32_vec(planar_f32, shape) {
                Ok(t) => Payload::Audio(Audio {
                    tensor: t,
                    sample_rate: sr,
                    channel_layout: AudioChannelLayout::from_channel_count(channels),
                    layout: AudioLayout::Planar,
                }),
                Err(err) => Payload::Error(err),
            }
        }
        Payload::Audio(mut audio) => {
            if let Some(sr) = target_sample_rate {
                audio.sample_rate = sr;
            }
            Payload::Audio(audio)
        }
        other => other,
    }
}

/// Converts a generic Tensor into a `Vec<f32>`, with optional normalization if integer dtype.
fn tensor_to_f32_vec(tensor: &Tensor, normalize: bool) -> Vec<f32> {
    match tensor.dtype {
        TensorDType::F32 => tensor.to_vec_f32(),
        TensorDType::U8 => {
            let bytes = tensor.to_contiguous_bytes();
            if normalize {
                bytes.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
            } else {
                bytes.iter().map(|&b| b as f32).collect()
            }
        }
        TensorDType::I32 => {
            let bytes = tensor.to_contiguous_bytes();
            let i32_slice: &[i32] = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const i32,
                    bytes.len() / std::mem::size_of::<i32>(),
                )
            };
            if normalize {
                i32_slice.iter().map(|&i| i as f32 / 2147483648.0).collect()
            } else {
                i32_slice.iter().map(|&i| i as f32).collect()
            }
        }
        _ => tensor.to_vec_f32(),
    }
}

/// Attempts to parse a binary buffer as a RIFF/WAVE file and decode samples to planar Float32.
fn try_parse_wav(bytes: &[u8], override_sr: Option<u32>) -> Option<Audio> {
    if bytes.len() < 44 {
        return None;
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }

    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut audio_format: u16 = 1; // 1 = PCM, 3 = IEEE Float, 0xFFFE = Extensible
    let mut data_offset: usize = 0;
    let mut data_length: usize = 0;

    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size =
            u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let chunk_data_start = offset + 8;

        if chunk_id == b"fmt " && chunk_data_start + 16 <= bytes.len() {
            audio_format = u16::from_le_bytes(
                bytes[chunk_data_start..chunk_data_start + 2]
                    .try_into()
                    .unwrap(),
            );
            channels = u16::from_le_bytes(
                bytes[chunk_data_start + 2..chunk_data_start + 4]
                    .try_into()
                    .unwrap(),
            );
            sample_rate = u32::from_le_bytes(
                bytes[chunk_data_start + 4..chunk_data_start + 8]
                    .try_into()
                    .unwrap(),
            );
            bits_per_sample = u16::from_le_bytes(
                bytes[chunk_data_start + 14..chunk_data_start + 16]
                    .try_into()
                    .unwrap(),
            );

            // Handle WAVE_FORMAT_EXTENSIBLE (format tag 0xFFFE)
            if audio_format == 0xFFFE && chunk_size >= 40 && chunk_data_start + 26 <= bytes.len() {
                let sub_format = u16::from_le_bytes(
                    bytes[chunk_data_start + 24..chunk_data_start + 26]
                        .try_into()
                        .unwrap(),
                );
                audio_format = sub_format;
            }
        } else if chunk_id == b"data" {
            data_offset = chunk_data_start;
            data_length = chunk_size.min(bytes.len().saturating_sub(chunk_data_start));
        }

        offset += 8 + chunk_size;
        // RIFF word alignment
        if !chunk_size.is_multiple_of(2) {
            offset += 1;
        }
    }

    if channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data_offset == 0 {
        return None;
    }

    let ch = channels as usize;
    let sr = override_sr.unwrap_or(sample_rate);
    let raw_samples_data = &bytes[data_offset..data_offset + data_length];

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    if bytes_per_sample == 0 {
        return None;
    }
    let total_samples = raw_samples_data.len() / bytes_per_sample;
    let samples_per_channel = total_samples / ch;
    if samples_per_channel == 0 {
        return None;
    }

    // Decode interleaved samples into planar Float32
    let mut planar = vec![0.0f32; ch * samples_per_channel];

    match (audio_format, bits_per_sample) {
        (1, 16) => {
            // 16-bit signed PCM
            planar
                .par_chunks_exact_mut(samples_per_channel)
                .enumerate()
                .for_each(|(c, plane)| {
                    for s in 0..samples_per_channel {
                        let idx = (s * ch + c) * 2;
                        if idx + 2 <= raw_samples_data.len() {
                            let val = i16::from_le_bytes([
                                raw_samples_data[idx],
                                raw_samples_data[idx + 1],
                            ]);
                            plane[s] = val as f32 / 32768.0;
                        }
                    }
                });
        }
        (1, 24) => {
            // 24-bit signed PCM
            planar
                .par_chunks_exact_mut(samples_per_channel)
                .enumerate()
                .for_each(|(c, plane)| {
                    for s in 0..samples_per_channel {
                        let idx = (s * ch + c) * 3;
                        if idx + 3 <= raw_samples_data.len() {
                            let b0 = raw_samples_data[idx] as u32;
                            let b1 = raw_samples_data[idx + 1] as u32;
                            let b2 = raw_samples_data[idx + 2] as u32;
                            let mut val = (b0 | (b1 << 8) | (b2 << 16)) as i32;
                            // Sign extend from 24-bit to 32-bit
                            if val & 0x00800000 != 0 {
                                val |= !0x00FFFFFF;
                            }
                            plane[s] = val as f32 / 8388608.0;
                        }
                    }
                });
        }
        (1, 32) => {
            // 32-bit signed integer PCM
            planar
                .par_chunks_exact_mut(samples_per_channel)
                .enumerate()
                .for_each(|(c, plane)| {
                    for s in 0..samples_per_channel {
                        let idx = (s * ch + c) * 4;
                        if idx + 4 <= raw_samples_data.len() {
                            let val = i32::from_le_bytes(
                                raw_samples_data[idx..idx + 4].try_into().unwrap(),
                            );
                            plane[s] = val as f32 / 2147483648.0;
                        }
                    }
                });
        }
        (3, 32) => {
            // 32-bit IEEE float
            planar
                .par_chunks_exact_mut(samples_per_channel)
                .enumerate()
                .for_each(|(c, plane)| {
                    for s in 0..samples_per_channel {
                        let idx = (s * ch + c) * 4;
                        if idx + 4 <= raw_samples_data.len() {
                            let val = f32::from_le_bytes(
                                raw_samples_data[idx..idx + 4].try_into().unwrap(),
                            );
                            plane[s] = val;
                        }
                    }
                });
        }
        (1, 8) => {
            // 8-bit unsigned PCM
            planar
                .par_chunks_exact_mut(samples_per_channel)
                .enumerate()
                .for_each(|(c, plane)| {
                    for s in 0..samples_per_channel {
                        let idx = s * ch + c;
                        if idx < raw_samples_data.len() {
                            let val = raw_samples_data[idx];
                            plane[s] = (val as f32 - 128.0) / 128.0;
                        }
                    }
                });
        }
        _ => return None,
    }

    let shape = if ch == 1 {
        vec![planar.len()]
    } else {
        vec![ch, planar.len() / ch]
    };
    let tensor = Tensor::from_f32_vec(planar, shape).ok()?;
    Some(Audio {
        tensor,
        sample_rate: sr,
        channel_layout: AudioChannelLayout::from_channel_count(ch),
        layout: AudioLayout::Planar,
    })
}

/// Decodes raw PCM byte buffer into planar Float32.
fn decode_raw_pcm_to_planar_f32(
    bytes: &[u8],
    channels: usize,
    dtype: &str,
    is_planar: bool,
    normalize: bool,
) -> Vec<f32> {
    match dtype.to_lowercase().as_str() {
        "f32" | "float" | "float32" => {
            if !bytes.len().is_multiple_of(4) {
                return Vec::new();
            }
            let total_samples = bytes.len() / 4;
            let samples_per_ch = total_samples / channels;
            if samples_per_ch == 0 {
                return Vec::new();
            }

            let f32_raw: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, total_samples)
            };

            if is_planar || channels == 1 {
                f32_raw[..channels * samples_per_ch].to_vec()
            } else {
                let mut planar = vec![0.0f32; channels * samples_per_ch];
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            plane[s] = f32_raw[s * channels + c];
                        }
                    });
                planar
            }
        }
        "i24" | "int24" => {
            if !bytes.len().is_multiple_of(3) {
                return Vec::new();
            }
            let total_samples = bytes.len() / 3;
            let samples_per_ch = total_samples / channels;
            if samples_per_ch == 0 {
                return Vec::new();
            }

            let mut planar = vec![0.0f32; channels * samples_per_ch];
            if is_planar {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let idx = (c * samples_per_ch + s) * 3;
                            let b0 = bytes[idx] as u32;
                            let b1 = bytes[idx + 1] as u32;
                            let b2 = bytes[idx + 2] as u32;
                            let mut val = (b0 | (b1 << 8) | (b2 << 16)) as i32;
                            if val & 0x00800000 != 0 {
                                val |= !0x00FFFFFF;
                            }
                            plane[s] = if normalize {
                                val as f32 / 8388608.0
                            } else {
                                val as f32
                            };
                        }
                    });
            } else {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let idx = (s * channels + c) * 3;
                            let b0 = bytes[idx] as u32;
                            let b1 = bytes[idx + 1] as u32;
                            let b2 = bytes[idx + 2] as u32;
                            let mut val = (b0 | (b1 << 8) | (b2 << 16)) as i32;
                            if val & 0x00800000 != 0 {
                                val |= !0x00FFFFFF;
                            }
                            plane[s] = if normalize {
                                val as f32 / 8388608.0
                            } else {
                                val as f32
                            };
                        }
                    });
            }
            planar
        }
        "i32" | "int32" => {
            if !bytes.len().is_multiple_of(4) {
                return Vec::new();
            }
            let total_samples = bytes.len() / 4;
            let samples_per_ch = total_samples / channels;
            if samples_per_ch == 0 {
                return Vec::new();
            }

            let mut planar = vec![0.0f32; channels * samples_per_ch];
            if is_planar {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let idx = (c * samples_per_ch + s) * 4;
                            let val = i32::from_le_bytes(bytes[idx..idx + 4].try_into().unwrap());
                            plane[s] = if normalize {
                                val as f32 / 2147483648.0
                            } else {
                                val as f32
                            };
                        }
                    });
            } else {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let idx = (s * channels + c) * 4;
                            let val = i32::from_le_bytes(bytes[idx..idx + 4].try_into().unwrap());
                            plane[s] = if normalize {
                                val as f32 / 2147483648.0
                            } else {
                                val as f32
                            };
                        }
                    });
            }
            planar
        }
        "u8" | "uint8" => {
            let total_samples = bytes.len();
            let samples_per_ch = total_samples / channels;
            if samples_per_ch == 0 {
                return Vec::new();
            }

            let mut planar = vec![0.0f32; channels * samples_per_ch];
            if is_planar {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let val = bytes[c * samples_per_ch + s];
                            plane[s] = if normalize {
                                (val as f32 - 128.0) / 128.0
                            } else {
                                val as f32
                            };
                        }
                    });
            } else {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let val = bytes[s * channels + c];
                            plane[s] = if normalize {
                                (val as f32 - 128.0) / 128.0
                            } else {
                                val as f32
                            };
                        }
                    });
            }
            planar
        }
        _ /* "i16" / "int16" */ => {
            if !bytes.len().is_multiple_of(2) {
                return Vec::new();
            }
            let total_samples = bytes.len() / 2;
            let samples_per_ch = total_samples / channels;
            if samples_per_ch == 0 {
                return Vec::new();
            }

            let mut planar = vec![0.0f32; channels * samples_per_ch];
            if is_planar {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let idx = (c * samples_per_ch + s) * 2;
                            let val = i16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
                            plane[s] = if normalize {
                                val as f32 / 32768.0
                            } else {
                                val as f32
                            };
                        }
                    });
            } else {
                planar
                    .par_chunks_exact_mut(samples_per_ch)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..samples_per_ch {
                            let idx = (s * channels + c) * 2;
                            let val = i16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
                            plane[s] = if normalize {
                                val as f32 / 32768.0
                            } else {
                                val as f32
                            };
                        }
                    });
            }
            planar
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    fn make_test_wav_16bit(channels: u16, sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let mut buf = Vec::new();
        let data_size = (samples.len() * 2) as u32;
        let byte_rate = sample_rate * channels as u32 * 2;
        let block_align = channels * 2;

        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_size).to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes()); // 16 bits

        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_to_audio_from_wav_bytes() {
        // 2 channels, 48000 Hz, 4 frames (8 samples total interleaved)
        let samples = vec![16384, -16384, 32767, -32768, 0, 0, 8192, -8192];
        let wav_data = make_test_wav_16bit(2, 48000, &samples);

        let payload = Payload::Data {
            buffer: core_types::RVec::from(wav_data),
        };

        let result = process(payload);
        if let Payload::Audio(audio) = result {
            assert_eq!(audio.sample_rate, 48000);
            assert_eq!(audio.channels(), 2);
            assert_eq!(audio.num_samples(), 4);
            let planar_f32 = audio.to_vec_f32();
            assert_eq!(planar_f32.len(), 8);
            // Channel 0 (left): 16384/32768 = 0.5, 32767/32768 ~= 0.999969, 0, 8192/32768 = 0.25
            assert!((planar_f32[0] - 0.5).abs() < 1e-4);
            assert!((planar_f32[1] - 1.0).abs() < 1e-3);
            assert_eq!(planar_f32[2], 0.0);
            assert!((planar_f32[3] - 0.25).abs() < 1e-4);

            // Channel 1 (right): -16384/32768 = -0.5, -32768/32768 = -1.0, 0, -8192/32768 = -0.25
            assert!((planar_f32[4] - (-0.5)).abs() < 1e-4);
            assert!((planar_f32[5] - (-1.0)).abs() < 1e-4);
            assert_eq!(planar_f32[6], 0.0);
            assert!((planar_f32[7] - (-0.25)).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Audio, got {:?}", result);
        }
    }

    #[test]
    fn test_to_audio_from_raw_pcm() {
        let raw_i16 = vec![32767i16, -32768i16]; // 1 stereo frame (left 1.0, right -1.0)
        let mut byte_vec = Vec::new();
        for s in raw_i16 {
            byte_vec.extend_from_slice(&s.to_le_bytes());
        }

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("channels"), RString::from("2")));
        named.push(Tuple2(RString::from("sample_rate"), RString::from("44100")));
        named.push(Tuple2(RString::from("dtype"), RString::from("i16")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Data {
                buffer: core_types::RVec::from(byte_vec),
            }),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Audio(audio) = result {
            assert_eq!(audio.sample_rate, 44100);
            assert_eq!(audio.channels(), 2);
            assert_eq!(audio.num_samples(), 1);
            let planar = audio.to_vec_f32();
            assert!((planar[0] - 1.0).abs() < 1e-3);
            assert!((planar[1] - (-1.0)).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Audio");
        }
    }

    #[test]
    fn test_to_audio_from_tensor() {
        let f32_samples = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let tensor = Tensor::from_f32_shape(&f32_samples, vec![2, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("sample_rate"), RString::from("96000")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Audio(audio) = result {
            assert_eq!(audio.sample_rate, 96000);
            assert_eq!(audio.channels(), 2);
            assert_eq!(audio.num_samples(), 3);
            assert_eq!(audio.to_vec_f32(), f32_samples);
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
