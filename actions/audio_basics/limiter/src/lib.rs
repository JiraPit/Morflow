use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, RString, Shape, ShapeResult, TensorDType};
use rayon::prelude::*;

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Audio
}

#[cfg_attr(not(morflow_pack), no_mangle)]
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

#[cfg_attr(not(morflow_pack), no_mangle)]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));
    let mut audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'limiter' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'limiter' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut ceiling_db = -0.1f32;
    let mut release_ms = 50.0f32;
    let mut mode = "brickwall".to_string();
    let mut drive = 1.0f32;
    let mut sample_rate = audio.sample_rate as f32;

    if let Some(args) = &args_opt {
        if let Some(c) = args
            .get_named("ceiling_db")
            .or_else(|| args.get_named("ceiling"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(v) = prepared.args.parse::<f32>(c) {
                ceiling_db = v;
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
        if let Some(m) = args.get_named("mode") {
            mode = m.to_lowercase();
        }
        if let Some(d) = args.get_named("drive") {
            if let Ok(v) = prepared.args.parse::<f32>(d) {
                drive = v;
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

    let params = LimiterParams::new(ceiling_db, release_ms, &mode, drive, sample_rate);

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

#[cfg_attr(not(morflow_pack), no_mangle)]
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
    fn test_limiter_brickwall() {
        let input_samples = vec![2.0f32, -3.0f32, 1.5f32, -0.2f32];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("ceiling_db"), RString::from("0.0")));
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
            for &s in out_slice {
                assert!(s.abs() <= 1.0 + 1e-5, "Sample exceeded ceiling: {}", s);
            }
        } else {
            panic!("Expected Payload::Audio");
        }
    }

    #[test]
    fn test_limiter_takes_the_ceiling_positionally() {
        let samples = vec![0.9f32, -0.8, 0.95, -0.2];
        let run = |positional: Vec<&str>, named: Vec<(&str, &str)>| {
            let audio = Audio::from_f32_planar(&samples, 1, 44100).unwrap();
            let mut named_args = core_types::RVec::new();
            for (name, value) in named {
                named_args.push(Tuple2(RString::from(name), RString::from(value)));
            }
            match process(Payload::WithArgs {
                payload: RBox::new(Payload::Audio(audio)),
                args: ActionArgs {
                    positional: positional.into_iter().map(RString::from).collect(),
                    named: named_args,
                },
            }) {
                Payload::Audio(out) => out.as_f32_slice().unwrap().to_vec(),
                _ => panic!("Expected Payload::Audio"),
            }
        };
        let by_name = run(vec![], vec![("ceiling_db", "-6.0")]);
        assert_eq!(run(vec!["-6.0"], vec![]), by_name);
        // -6 dBFS is about 0.501, well below the default -0.1 dBFS ceiling.
        assert!(by_name.iter().all(|s| s.abs() <= 0.502));
        assert_ne!(run(vec![], vec![]), by_name);
    }
}
