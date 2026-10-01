use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, RString, Shape, ShapeResult, TensorDType};
use rayon::prelude::*;
use std::f32::consts::PI;

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
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl BiquadCoeffs {
    fn new(filter_type: &str, freq: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let f0 = freq.clamp(10.0, sample_rate * 0.499);
        let q = q.max(0.01);
        let w0 = 2.0 * PI * f0 / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a_linear = 10.0f32.powf(gain_db / 40.0);

        let (b0_raw, b1_raw, b2_raw, a0_raw, a1_raw, a2_raw) = match filter_type {
            "highpass" | "hp" => {
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            "bandpass" | "bp" => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            "notch" => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            "peaking" | "peak" | "eq" => {
                let b0 = 1.0 + alpha * a_linear;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a_linear;
                let a0 = 1.0 + alpha / a_linear;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a_linear;
                (b0, b1, b2, a0, a1, a2)
            }
            _ /* lowpass */ => {
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        Self {
            b0: b0_raw / a0_raw,
            b1: b1_raw / a0_raw,
            b2: b2_raw / a0_raw,
            a1: a1_raw / a0_raw,
            a2: a2_raw / a0_raw,
        }
    }

    /// Processes a single contiguous audio channel in-place with Direct Form II Transposed.
    fn process_channel(&self, samples: &mut [f32]) {
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;

        for x in samples.iter_mut() {
            let input = *x;
            let output = self.b0 * input + s1;
            s1 = self.b1 * input - self.a1 * output + s2;
            s2 = self.b2 * input - self.a2 * output;
            *x = output;
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
            return Payload::Error(RString::from(
                "Action 'biquad_filter' requires Payload::Audio",
            ));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'biquad_filter' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut filter_type = "lowpass".to_string();
    let mut freq = 1000.0f32;
    let mut q = 0.707f32;
    let mut gain_db = 0.0f32;
    let mut sample_rate = audio.sample_rate as f32;

    if let Some(args) = &args_opt {
        if let Some(t) = args.get_named("type") {
            filter_type = t.to_lowercase();
        }
        if let Some(f_str) = args
            .get_named("freq")
            .or_else(|| args.get_named("cutoff_hz"))
        {
            if let Ok(f) = prepared.args.parse::<f32>(f_str) {
                freq = f;
            }
        }
        if let Some(q_str) = args.get_named("q") {
            if let Ok(val) = prepared.args.parse::<f32>(q_str) {
                q = val;
            }
        }
        if let Some(g_str) = args.get_named("gain_db").or_else(|| args.get_named("gain")) {
            if let Ok(g) = prepared.args.parse::<f32>(g_str) {
                gain_db = g;
            }
        }
        if let Some(sr_str) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
        {
            if let Ok(sr) = prepared.args.parse::<f32>(sr_str) {
                sample_rate = sr;
            }
        }
    }

    let coeffs = BiquadCoeffs::new(&filter_type, freq, q, gain_db, sample_rate);

    let channel_len = if audio.channels() > 1 {
        audio.num_samples()
    } else {
        0
    };
    let samples = audio.tensor.as_f32_slice_mut();

    if channel_len > 0 {
        samples
            .par_chunks_mut(channel_len)
            .for_each(|ch| coeffs.process_channel(ch));
    } else {
        coeffs.process_channel(samples);
    }

    Payload::Audio(audio)
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
    fn test_lowpass_filter() {
        // High-frequency alternating signal [1.0, -1.0, 1.0, -1.0, ...]
        let input_samples: Vec<f32> = (0..100usize)
            .map(|i| if i.is_multiple_of(2) { 1.0f32 } else { -1.0f32 })
            .collect();
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("type"), RString::from("lowpass")));
        named.push(Tuple2(RString::from("freq"), RString::from("200.0")));
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
            // Nyquist frequency at 22050Hz attenuated heavily by 200Hz lowpass
            let last_val = out_slice[99].abs();
            assert!(
                last_val < 0.1,
                "High freq should be attenuated, got {}",
                last_val
            );
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
