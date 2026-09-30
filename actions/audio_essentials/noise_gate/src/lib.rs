use core_types::{
    ActionArgs, DataType, GetShapeFn, Payload, RString, Shape, ShapeResult, TensorDType,
};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Audio
}

fn shape_impl(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub static MORFLOW_SHAPE_ABI: u32 = core_types::SHAPE_ABI_VERSION;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[derive(Clone, Copy)]
struct NoiseGateParams {
    threshold_db: f32,
    ratio: f32,
    alpha_attack: f32,
    alpha_release: f32,
    hold_samples: usize,
}

impl NoiseGateParams {
    fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        hold_ms: f32,
        release_ms: f32,
        sample_rate: f32,
    ) -> Self {
        let att_sec = (attack_ms * 0.001).max(0.0001);
        let rel_sec = (release_ms * 0.001).max(0.001);
        let alpha_attack = (-1.0 / (att_sec * sample_rate)).exp();
        let alpha_release = (-1.0 / (rel_sec * sample_rate)).exp();
        let hold_samples = ((hold_ms * 0.001) * sample_rate) as usize;

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            alpha_attack,
            alpha_release,
            hold_samples,
        }
    }

    fn process_channel(&self, samples: &mut [f32]) {
        let mut envelope_gain = 0.0f32; // 0 = closed, 1 = open
        let mut hold_counter = 0usize;

        for x in samples.iter_mut() {
            let input_mag = x.abs();
            let input_db = if input_mag > 1e-6 {
                20.0 * input_mag.log10()
            } else {
                -120.0
            };

            let target_gain = if input_db >= self.threshold_db {
                hold_counter = self.hold_samples;
                1.0f32
            } else if hold_counter > 0 {
                hold_counter -= 1;
                1.0f32
            } else {
                // Downward expansion below threshold
                let reduction_db = (self.threshold_db - input_db) * (self.ratio - 1.0);
                10.0f32.powf(-reduction_db / 20.0).clamp(0.0, 1.0)
            };

            // Ballistics smoothing
            if target_gain > envelope_gain {
                envelope_gain =
                    self.alpha_attack * envelope_gain + (1.0 - self.alpha_attack) * target_gain;
            } else {
                envelope_gain =
                    self.alpha_release * envelope_gain + (1.0 - self.alpha_release) * target_gain;
            }

            *x *= envelope_gain;
        }
    }
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'noise_gate' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'noise_gate' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut threshold_db = -40.0f32;
    let mut ratio = 10.0f32;
    let mut attack_ms = 2.0f32;
    let mut hold_ms = 10.0f32;
    let mut release_ms = 50.0f32;
    let mut sample_rate = audio.sample_rate as f32;

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
        if let Some(h) = args.get_named("hold_ms").or_else(|| args.get_named("hold")) {
            if let Ok(v) = h.parse::<f32>() {
                hold_ms = v;
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
        if let Some(sr) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
        {
            if let Ok(v) = sr.parse::<f32>() {
                sample_rate = v;
            }
        }
    }

    let params = NoiseGateParams::new(
        threshold_db,
        ratio,
        attack_ms,
        hold_ms,
        release_ms,
        sample_rate,
    );

    let channel_len = if audio.channels() > 1 {
        audio.num_samples()
    } else {
        0
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_noise_gate_attenuation() {
        // Very low signal (-60 dBFS approx 0.001) below -30dB threshold
        let input_samples = vec![0.001f32; 1000];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(
            RString::from("threshold_db"),
            RString::from("-30.0"),
        ));
        named.push(Tuple2(RString::from("ratio"), RString::from("10.0")));
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
            let out_slice: &[f32] = out_aud.as_f32_slice().unwrap();
            let end_sample = out_slice[999];
            assert!(
                end_sample < 0.0001,
                "Expected noise gate suppression, got {}",
                end_sample
            );
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
