use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, RString, Shape, ShapeResult, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Audio
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if !matches!(input.rank(), 1 | 2) {
            return Err("Audio operations require rank 1 or 2".into());
        }
        Ok(input)
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
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
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));
    let mut audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'compressor' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'compressor' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut threshold_db = -12.0f32;
    let mut ratio = 4.0f32;
    let mut attack_ms = 10.0f32;
    let mut release_ms = 100.0f32;
    let mut knee_db = 2.0f32;
    let mut makeup_db = 0.0f32;
    let mut sample_rate = audio.sample_rate as f32;

    if let Some(args) = &args_opt {
        if let Some(t) = args
            .get_named("threshold_db")
            .or_else(|| args.get_named("threshold"))
        {
            if let Ok(v) = prepared.args.parse::<f32>(t) {
                threshold_db = v;
            }
        }
        if let Some(r) = args.get_named("ratio") {
            if let Ok(v) = prepared.args.parse::<f32>(r) {
                ratio = v;
            }
        }
        if let Some(a) = args
            .get_named("attack_ms")
            .or_else(|| args.get_named("attack"))
        {
            if let Ok(v) = prepared.args.parse::<f32>(a) {
                attack_ms = v;
            }
        }
        if let Some(rel) = args
            .get_named("release_ms")
            .or_else(|| args.get_named("release"))
        {
            if let Ok(v) = prepared.args.parse::<f32>(rel) {
                release_ms = v;
            }
        }
        if let Some(k) = args.get_named("knee_db").or_else(|| args.get_named("knee")) {
            if let Ok(v) = prepared.args.parse::<f32>(k) {
                knee_db = v;
            }
        }
        if let Some(m) = args
            .get_named("makeup_db")
            .or_else(|| args.get_named("makeup"))
        {
            if let Ok(v) = prepared.args.parse::<f32>(m) {
                makeup_db = v;
            }
        }
        if let Some(sr) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
        {
            if let Ok(v) = prepared.args.parse::<f32>(sr) {
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

#[no_mangle]
pub extern "C" fn get_action_abi_version() -> u32 {
    core_types::shapecheck::ACTION_ABI_VERSION
}
#[no_mangle]
pub extern "C" fn get_action_abi_layout() -> *const core_types::abi_stable::type_layout::TypeLayout
{
    <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT
}
#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_compressor_gain_reduction() {
        // High level signal at 1.0 (0 dBFS) with threshold -20dB
        let input_samples = vec![1.0f32; 1000];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

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
            payload: RBox::new(Payload::Audio(audio)),
            args,
        };

        let result = process(payload);
        if let Payload::Audio(out_aud) = result {
            let out_slice: &[f32] = out_aud.as_f32_slice().unwrap();
            // Gain should be significantly reduced at steady state (< 0.5)
            let end_sample = out_slice[999];
            assert!(
                end_sample < 0.5,
                "Expected gain reduction, got {}",
                end_sample
            );
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
