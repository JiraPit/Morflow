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
    let (inner_payload, args_opt) = payload.take_payload_and_args();
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
}
