use core_types::{DataType, Payload, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    // 1. Resolve gain multiplier from arguments
    let mut multiplier = 1.0f32;
    if let Some(args) = payload.args() {
        if let Some(lin_str) = args
            .get_named("linear")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(lin) = lin_str.parse::<f32>() {
                multiplier = lin;
            }
        }
        if let Some(db_str) = args.get_named("db") {
            if let Ok(db) = db_str.parse::<f32>() {
                multiplier = 10.0f32.powf(db / 20.0);
            }
        }
    }

    match payload.unwrap_payload() {
        Payload::Audio(audio) if audio.dtype() == TensorDType::F32 => {
            let mut bytes = audio.tensor.to_contiguous_bytes();
            let samples: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            samples.par_iter_mut().for_each(|s| *s *= multiplier);

            let out_tensor = Tensor::from_f32_shape(samples, audio.tensor.shape.to_vec())
                .unwrap_or_else(|_| audio.tensor.clone());
            let out_audio = core_types::Audio {
                tensor: out_tensor,
                sample_rate: audio.sample_rate,
                channel_layout: audio.channel_layout,
                layout: audio.layout,
            };
            Payload::Audio(out_audio)
        }
        Payload::Tensor(tensor) if tensor.dtype == TensorDType::F32 => {
            let mut bytes = tensor.to_contiguous_bytes();
            let samples: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            samples.par_iter_mut().for_each(|s| *s *= multiplier);

            let out_tensor = Tensor::from_f32_shape(samples, tensor.shape.to_vec())
                .unwrap_or_else(|_| tensor.clone());
            Payload::Tensor(out_tensor)
        }
        Payload::Data { buffer } => {
            // Raw bytes fallback (if treating data as f32 stream)
            if buffer.len() % 4 == 0 && !buffer.is_empty() {
                let mut out_buf = buffer.to_vec();
                let samples: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_buf.as_mut_ptr() as *mut f32,
                        out_buf.len() / 4,
                    )
                };
                samples.par_iter_mut().for_each(|s| *s *= multiplier);
                Payload::Data {
                    buffer: core_types::RVec::from(out_buf),
                }
            } else {
                payload.clone()
            }
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_gain_linear() {
        let input_samples = vec![0.5f32, -0.5f32, 1.0f32, -1.0f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 4]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("linear"), RString::from("2.0")));
        let args = ActionArgs {
            positional: core_types::RVec::new(),
            named,
        };

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args,
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            let out_slice: &[f32] = out_t.as_f32_slice().unwrap();
            assert_eq!(out_slice, &[1.0, -1.0, 2.0, -2.0]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }

    #[test]
    fn test_gain_db() {
        let input_samples = vec![1.0f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 1]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("db"), RString::from("6.0206")));
        let args = ActionArgs {
            positional: core_types::RVec::new(),
            named,
        };

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args,
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            let out_slice: &[f32] = out_t.as_f32_slice().unwrap();
            assert!((out_slice[0] - 2.0).abs() < 1e-3);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }

    #[test]
    fn test_gain_audio_payload() {
        let input_samples = vec![0.5f32, -0.5f32, 1.0f32, -1.0f32];
        let audio = core_types::Audio::from_f32_planar(&input_samples, 2, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("linear"), RString::from("3.0")));
        let args = ActionArgs {
            positional: core_types::RVec::new(),
            named,
        };

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Audio(audio)),
            args,
        };

        let result = process(payload);
        if let Payload::Audio(out_aud) = result {
            assert_eq!(out_aud.sample_rate, 44100);
            assert_eq!(out_aud.channels(), 2);
            let out_slice: &[f32] = out_aud.as_f32_slice().unwrap();
            assert_eq!(out_slice, &[1.5, -1.5, 3.0, -3.0]);
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
