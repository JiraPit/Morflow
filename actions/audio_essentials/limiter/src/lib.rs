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

#[derive(Clone, Copy)]
struct LimiterParams {
    ceiling: f32,
    release_coeff: f32,
    mode_soft_clip: bool,
    drive: f32,
}

impl LimiterParams {
    fn new(ceiling_db: f32, release_ms: f32, mode: &str, drive: f32, sample_rate: f32) -> Self {
        let ceiling = 10.0f32.powf(ceiling_db / 20.0);
        let rel_sec = (release_ms * 0.001).max(0.001);
        let release_coeff = (-1.0 / (rel_sec * sample_rate)).exp();
        let mode_soft_clip = mode == "soft_clip" || mode == "clip" || mode == "saturate";

        Self {
            ceiling,
            release_coeff,
            mode_soft_clip,
            drive: drive.max(0.1),
        }
    }

    fn process_channel(&self, samples: &mut [f32]) {
        if self.mode_soft_clip {
            // Hyperbolic tangent soft saturation curve
            let c = self.ceiling;
            let d = self.drive;
            for x in samples.iter_mut() {
                let scaled = (*x * d) / c;
                *x = scaled.tanh() * c;
            }
        } else {
            // Fast peak limiter
            let mut envelope = 0.0f32;
            let c = self.ceiling;

            for x in samples.iter_mut() {
                let peak = x.abs();
                if peak > envelope {
                    envelope = peak;
                } else {
                    envelope = self.release_coeff * envelope + (1.0 - self.release_coeff) * peak;
                }

                let gain = if envelope > c { c / envelope } else { 1.0 };
                *x *= gain;

                // Hard safety clamp at ceiling
                *x = x.clamp(-c, c);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut ceiling_db = -0.1f32;
    let mut release_ms = 50.0f32;
    let mut mode = "brickwall".to_string();
    let mut drive = 1.0f32;
    let mut sample_rate = 44100.0f32;

    if let Payload::Audio(audio) = payload.unwrap_payload() {
        sample_rate = audio.sample_rate as f32;
    }

    if let Some(args) = payload.args() {
        if let Some(c) = args
            .get_named("ceiling_db")
            .or_else(|| args.get_named("ceiling"))
        {
            if let Ok(v) = c.parse::<f32>() {
                ceiling_db = v;
            }
        }
        if let Some(rel) = args
            .get_named("release_ms")
            .or_else(|| args.get_named("release"))
        {
            if let Ok(v) = rel.parse::<f32>() {
                release_ms = v;
            }
        }
        if let Some(m) = args.get_named("mode") {
            mode = m.to_lowercase();
        }
        if let Some(d) = args.get_named("drive") {
            if let Ok(v) = d.parse::<f32>() {
                drive = v;
            }
        }
        if let Some(sr) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
        {
            if let Ok(v) = sr.parse::<f32>() {
                sample_rate = v;
            }
        }
    }

    let params = LimiterParams::new(ceiling_db, release_ms, &mode, drive, sample_rate);

    match payload.unwrap_payload() {
        Payload::Audio(audio) if audio.dtype() == TensorDType::F32 => {
            let mut bytes = audio.tensor.to_contiguous_bytes();
            let samples: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            let shape = audio.tensor.shape.as_slice();
            if shape.len() == 2 {
                let channel_len = shape[1];
                if channel_len > 0 {
                    samples
                        .par_chunks_mut(channel_len)
                        .for_each(|ch| params.process_channel(ch));
                }
            } else {
                params.process_channel(samples);
            }

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

            let shape = tensor.shape.as_slice();
            if shape.len() == 2 {
                let channel_len = shape[1];
                if channel_len > 0 {
                    samples
                        .par_chunks_mut(channel_len)
                        .for_each(|ch| params.process_channel(ch));
                }
            } else {
                params.process_channel(samples);
            }

            let out_tensor = Tensor::from_f32_shape(samples, tensor.shape.to_vec())
                .unwrap_or_else(|_| tensor.clone());
            Payload::Tensor(out_tensor)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_limiter_brickwall() {
        let input_samples = vec![2.0f32, -3.0f32, 1.5f32, -0.2f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 4]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("ceiling_db"), RString::from("0.0")));
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
            for &s in out_slice {
                assert!(s.abs() <= 1.0 + 1e-5, "Sample exceeded ceiling: {}", s);
            }
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
