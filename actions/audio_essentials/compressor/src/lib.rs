use core_types::{DataType, Payload, TensorDType};
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
struct CompressorParams {
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    alpha_attack: f32,
    alpha_release: f32,
    makeup_linear: f32,
}

impl CompressorParams {
    fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
        makeup_db: f32,
        sample_rate: f32,
    ) -> Self {
        let att_sec = (attack_ms * 0.001).max(0.0001);
        let rel_sec = (release_ms * 0.001).max(0.001);
        let alpha_attack = (-1.0 / (att_sec * sample_rate)).exp();
        let alpha_release = (-1.0 / (rel_sec * sample_rate)).exp();
        let makeup_linear = 10.0f32.powf(makeup_db / 20.0);

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            knee_db: knee_db.max(0.0),
            alpha_attack,
            alpha_release,
            makeup_linear,
        }
    }

    /// Computes static characteristic gain reduction in dB for a given input level in dB.
    #[inline]
    fn gain_computer(&self, x_db: f32) -> f32 {
        let t = self.threshold_db;
        let r = self.ratio;
        let w = self.knee_db;

        if 2.0 * (x_db - t) < -w {
            // Below knee: 0 dB reduction
            0.0
        } else if 2.0 * (x_db - t).abs() <= w {
            // Inside knee: quadratic interpolation
            let delta = x_db - t + w * 0.5;
            (1.0 / r - 1.0) * (delta * delta) / (2.0 * w)
        } else {
            // Above knee: linear compression
            (1.0 / r - 1.0) * (x_db - t)
        }
    }

    fn process_channel(&self, samples: &mut [f32]) {
        let mut envelope_db = -96.0f32;

        for x in samples.iter_mut() {
            let input_mag = x.abs().max(1e-6);
            let input_db = 20.0 * input_mag.log10();

            // Gain computer
            let target_gr_db = self.gain_computer(input_db);

            // Ballistics smoothing
            if target_gr_db < envelope_db {
                envelope_db =
                    self.alpha_attack * envelope_db + (1.0 - self.alpha_attack) * target_gr_db;
            } else {
                envelope_db =
                    self.alpha_release * envelope_db + (1.0 - self.alpha_release) * target_gr_db;
            }

            let gr_linear = 10.0f32.powf(envelope_db / 20.0);
            *x *= gr_linear * self.makeup_linear;
        }
    }
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut threshold_db = -12.0f32;
    let mut ratio = 4.0f32;
    let mut attack_ms = 10.0f32;
    let mut release_ms = 100.0f32;
    let mut knee_db = 2.0f32;
    let mut makeup_db = 0.0f32;
    let mut sample_rate = 44100.0f32;

    if let Payload::Audio(audio) = &inner_payload {
        sample_rate = audio.sample_rate as f32;
    }

    if let Some(args) = &args_opt {
        if let Some(t) = args
            .get_named("threshold_db")
            .or_else(|| args.get_named("threshold"))
        {
            if let Ok(v) = t.parse::<f32>() {
                threshold_db = v;
            }
        }
        if let Some(r) = args.get_named("ratio") {
            if let Ok(v) = r.parse::<f32>() {
                ratio = v;
            }
        }
        if let Some(a) = args
            .get_named("attack_ms")
            .or_else(|| args.get_named("attack"))
        {
            if let Ok(v) = a.parse::<f32>() {
                attack_ms = v;
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
        if let Some(k) = args.get_named("knee_db").or_else(|| args.get_named("knee")) {
            if let Ok(v) = k.parse::<f32>() {
                knee_db = v;
            }
        }
        if let Some(m) = args
            .get_named("makeup_db")
            .or_else(|| args.get_named("makeup"))
        {
            if let Ok(v) = m.parse::<f32>() {
                makeup_db = v;
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

    let params = CompressorParams::new(
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        knee_db,
        makeup_db,
        sample_rate,
    );

    match inner_payload {
        Payload::Audio(mut audio) if audio.dtype() == TensorDType::F32 => {
            let shape = audio.tensor.shape.as_slice();
            let channel_len = if shape.len() == 2 { shape[1] } else { 0 };
            let samples = audio.tensor.as_f32_slice_mut();

            if channel_len > 0 {
                samples
                    .par_chunks_mut(channel_len)
                    .for_each(|ch| params.process_channel(ch));
            } else {
                params.process_channel(samples);
            }

            Payload::Audio(audio)
        }
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let shape = tensor.shape.as_slice();
            let channel_len = if shape.len() == 2 { shape[1] } else { 0 };
            let samples = tensor.as_f32_slice_mut();

            if channel_len > 0 {
                samples
                    .par_chunks_mut(channel_len)
                    .for_each(|ch| params.process_channel(ch));
            } else {
                params.process_channel(samples);
            }

            Payload::Tensor(tensor)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_compressor_gain_reduction() {
        // High level signal at 1.0 (0 dBFS) with threshold -20dB
        let input_samples = vec![1.0f32; 1000];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 1000]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(
            RString::from("threshold_db"),
            RString::from("-20.0"),
        ));
        named.push(Tuple2(RString::from("ratio"), RString::from("4.0")));
        named.push(Tuple2(RString::from("attack_ms"), RString::from("1.0")));
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
            // Gain should be significantly reduced at steady state (< 0.5)
            let end_sample = out_slice[999];
            assert!(
                end_sample < 0.5,
                "Expected gain reduction, got {}",
                end_sample
            );
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
