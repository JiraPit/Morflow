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
struct DelayParams {
    delay_samples: usize,
    feedback: f32,
    mix: f32,
}

impl DelayParams {
    fn new(time_ms: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        let delay_samples = ((time_ms.max(0.1) * 0.001) * sample_rate).round() as usize;
        Self {
            delay_samples: delay_samples.max(1),
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
        }
    }

    fn process_channel(&self, samples: &mut [f32]) {
        let buffer_size = self.delay_samples + 1;
        let mut ring_buffer = vec![0.0f32; buffer_size];
        let mut write_pos = 0usize;

        let dry_gain = 1.0 - self.mix;
        let wet_gain = self.mix;

        for x in samples.iter_mut() {
            let dry = *x;

            let read_pos = (write_pos + buffer_size - self.delay_samples) % buffer_size;
            let delayed = ring_buffer[read_pos];

            ring_buffer[write_pos] = dry + delayed * self.feedback;
            write_pos = (write_pos + 1) % buffer_size;

            *x = dry * dry_gain + delayed * wet_gain;
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
            return Payload::Error(RString::from("Action 'delay' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'delay' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut time_ms = 120.0f32;
    let mut feedback = 0.35f32;
    let mut mix = 0.3f32;
    let mut sample_rate = audio.sample_rate as f32;

    if let Some(args) = &args_opt {
        if let Some(t) = args
            .get_named("time_ms")
            .or_else(|| args.get_named("time"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(v) = prepared.args.parse::<f32>(t) {
                time_ms = v;
            }
        }
        if let Some(fb) = args.get_named("feedback") {
            if let Ok(v) = prepared.args.parse::<f32>(fb) {
                feedback = v;
            }
        }
        if let Some(m) = args.get_named("mix").or_else(|| args.get_named("wet")) {
            if let Ok(v) = prepared.args.parse::<f32>(m) {
                mix = v;
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

    let params = DelayParams::new(time_ms, feedback, mix, sample_rate);

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
    fn test_delay_echo() {
        // Single impulse at index 0 [1.0, 0.0, 0.0, 0.0]
        let input_samples = vec![1.0f32, 0.0f32, 0.0f32, 0.0f32];
        let audio = Audio::from_f32_planar(&input_samples, 1, 1000).unwrap();

        // Delay 2 samples at 1000 Hz = 2 ms
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("time_ms"), RString::from("2.0")));
        named.push(Tuple2(RString::from("feedback"), RString::from("0.0")));
        named.push(Tuple2(RString::from("mix"), RString::from("0.5")));
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
            // Dry at index 0: 0.5 * 1.0 = 0.5
            assert!((out_slice[0] - 0.5).abs() < 1e-4);
            // Echo at index 2 (2 samples later): 0.5 * 1.0 = 0.5
            assert!((out_slice[2] - 0.5).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
