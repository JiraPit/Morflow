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
    let mut target_layout = "interleaved";
    let mut clip = true;

    if let Some(args) = payload.args() {
        if let Some(dt) = args
            .get_named("dtype")
            .or_else(|| args.get_named("format"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            target_dtype = dt;
        }
        if let Some(lay) = args.get_named("layout") {
            target_layout = lay;
        }
        if let Some(clip_str) = args.get_named("clip") {
            clip = !matches!(
                clip_str.to_lowercase().as_str(),
                "false" | "0" | "no" | "off"
            );
        }
    }

    let is_interleaved = target_layout.eq_ignore_ascii_case("interleaved")
        || target_layout.eq_ignore_ascii_case("packed");

    let (planar_f32, channels, num_samples) = match payload.unwrap_payload() {
        Payload::Audio(audio) => {
            let ch = audio.channels().max(1);
            let samples = audio.num_samples();
            let f32_vec = audio.to_vec_f32();
            (f32_vec, ch, samples)
        }
        Payload::Tensor(tensor) => {
            let shape = tensor.shape.as_slice();
            let (ch, samples) = match shape.len() {
                1 => (1, shape[0]),
                2 => (shape[0], shape[1]),
                _ => (1, tensor.num_elements()),
            };
            let f32_vec = if tensor.dtype == TensorDType::F32 {
                tensor.to_vec_f32()
            } else {
                let bytes = tensor.to_contiguous_bytes();
                bytes.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
            };
            (f32_vec, ch, samples)
        }
        Payload::Data { buffer } => {
            return Payload::Data {
                buffer: buffer.clone(),
            };
        }
        other => return other.clone(),
    };

    if num_samples == 0 || channels == 0 {
        return Payload::Data { buffer: RVec::new() };
    }

    let pcm_bytes = encode_planar_f32_to_pcm(
        &planar_f32,
        channels,
        num_samples,
        target_dtype,
        is_interleaved,
        clip,
    );

    Payload::Data {
        buffer: RVec::from(pcm_bytes),
    }
}

/// Encodes planar Float32 samples into raw PCM bytes according to the target format and layout.
pub fn encode_planar_f32_to_pcm(
    planar_f32: &[f32],
    channels: usize,
    num_samples: usize,
    dtype: &str,
    is_interleaved: bool,
    clip: bool,
) -> Vec<u8> {
    let clamp_sample = |s: f32| -> f32 {
        if clip {
            s.clamp(-1.0, 1.0)
        } else {
            s
        }
    };

    match dtype.to_lowercase().as_str() {
        "f32" | "float" | "float32" => {
            let mut out = vec![0u8; channels * num_samples * 4];
            if !is_interleaved || channels == 1 {
                let f32_out: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        out.as_mut_ptr() as *mut f32,
                        channels * num_samples,
                    )
                };
                f32_out
                    .par_iter_mut()
                    .zip(planar_f32[..channels * num_samples].par_iter())
                    .for_each(|(dst, &src)| {
                        *dst = clamp_sample(src);
                    });
            } else {
                out.par_chunks_exact_mut(channels * 4)
                    .enumerate()
                    .for_each(|(s, frame)| {
                        for c in 0..channels {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let bytes = val.to_le_bytes();
                            frame[c * 4..c * 4 + 4].copy_from_slice(&bytes);
                        }
                    });
            }
            out
        }
        "i24" | "int24" => {
            let mut out = vec![0u8; channels * num_samples * 3];
            if !is_interleaved || channels == 1 {
                out.par_chunks_exact_mut(num_samples * 3)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..num_samples {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let i24 = (val * 8388607.0).round().clamp(-8388608.0, 8388607.0) as i32;
                            let b0 = (i24 & 0xFF) as u8;
                            let b1 = ((i24 >> 8) & 0xFF) as u8;
                            let b2 = ((i24 >> 16) & 0xFF) as u8;
                            plane[s * 3] = b0;
                            plane[s * 3 + 1] = b1;
                            plane[s * 3 + 2] = b2;
                        }
                    });
            } else {
                out.par_chunks_exact_mut(channels * 3)
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
            out
        }
        "i32" | "int32" => {
            let mut out = vec![0u8; channels * num_samples * 4];
            if !is_interleaved || channels == 1 {
                out.par_chunks_exact_mut(num_samples * 4)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..num_samples {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let i32_val = (val * 2147483647.0).round().clamp(-2147483648.0, 2147483647.0) as i32;
                            plane[s * 4..s * 4 + 4].copy_from_slice(&i32_val.to_le_bytes());
                        }
                    });
            } else {
                out.par_chunks_exact_mut(channels * 4)
                    .enumerate()
                    .for_each(|(s, frame)| {
                        for c in 0..channels {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let i32_val = (val * 2147483647.0).round().clamp(-2147483648.0, 2147483647.0) as i32;
                            frame[c * 4..c * 4 + 4].copy_from_slice(&i32_val.to_le_bytes());
                        }
                    });
            }
            out
        }
        "u8" | "uint8" => {
            let mut out = vec![0u8; channels * num_samples];
            if !is_interleaved || channels == 1 {
                out.par_chunks_exact_mut(num_samples)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..num_samples {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let u8_val = ((val * 127.0).round().clamp(-128.0, 127.0) + 128.0) as u8;
                            plane[s] = u8_val;
                        }
                    });
            } else {
                out.par_chunks_exact_mut(channels)
                    .enumerate()
                    .for_each(|(s, frame)| {
                        for c in 0..channels {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let u8_val = ((val * 127.0).round().clamp(-128.0, 127.0) + 128.0) as u8;
                            frame[c] = u8_val;
                        }
                    });
            }
            out
        }
        _ /* "i16" / "int16" */ => {
            let mut out = vec![0u8; channels * num_samples * 2];
            if !is_interleaved || channels == 1 {
                out.par_chunks_exact_mut(num_samples * 2)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for s in 0..num_samples {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let i16_val = (val * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
                            plane[s * 2..s * 2 + 2].copy_from_slice(&i16_val.to_le_bytes());
                        }
                    });
            } else {
                out.par_chunks_exact_mut(channels * 2)
                    .enumerate()
                    .for_each(|(s, frame)| {
                        for c in 0..channels {
                            let val = clamp_sample(planar_f32[c * num_samples + s]);
                            let i16_val = (val * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
                            frame[c * 2..c * 2 + 2].copy_from_slice(&i16_val.to_le_bytes());
                        }
                    });
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_to_pcm_i16_interleaved() {
        // 2 channels, 2 samples each: Left = [1.0, -1.0], Right = [0.5, -0.5]
        let planar = vec![1.0f32, -1.0, 0.5, -0.5];
        let audio = Audio::from_f32_planar(&planar, 2, 44100).unwrap();

        let payload = Payload::Audio(audio);
        let result = process(payload);

        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.len(), 8); // 2 samples * 2 channels * 2 bytes
            let s0_left = i16::from_le_bytes([buffer[0], buffer[1]]);
            let s0_right = i16::from_le_bytes([buffer[2], buffer[3]]);
            let s1_left = i16::from_le_bytes([buffer[4], buffer[5]]);
            let s1_right = i16::from_le_bytes([buffer[6], buffer[7]]);

            assert_eq!(s0_left, 32767);
            assert_eq!(s0_right, 16384);
            assert_eq!(s1_left, -32767);
            assert_eq!(s1_right, -16384);
        } else {
            panic!("Expected Payload::Data");
        }
    }

    #[test]
    fn test_to_pcm_f32_planar() {
        let planar = vec![0.5f32, -0.5f32];
        let audio = Audio::from_f32_planar(&planar, 1, 48000).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dtype"), RString::from("f32")));
        named.push(Tuple2(RString::from("layout"), RString::from("planar")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Audio(audio)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.len(), 8);
            let f0 = f32::from_le_bytes(buffer[0..4].try_into().unwrap());
            let f1 = f32::from_le_bytes(buffer[4..8].try_into().unwrap());
            assert_eq!(f0, 0.5);
            assert_eq!(f1, -0.5);
        } else {
            panic!("Expected Payload::Data");
        }
    }
}
