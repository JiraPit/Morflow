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

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));
    let mut audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'gain' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'gain' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    // 1. Resolve gain multiplier from arguments
    let mut multiplier = 1.0f32;
    if let Some(args) = &args_opt {
        if let Some(lin_str) = args
            .get_named("linear")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(lin) = prepared.args.parse::<f32>(lin_str) {
                multiplier = lin;
            }
        }
        if let Some(db_str) = args.get_named("db") {
            if let Ok(db) = prepared.args.parse::<f32>(db_str) {
                multiplier = 10.0f32.powf(db / 20.0);
            }
        }
    }

    let samples = audio.tensor.as_f32_slice_mut();
    samples.par_iter_mut().for_each(|s| *s *= multiplier);
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
    fn test_gain_linear() {
        let input_samples = vec![0.5f32, -0.5f32, 1.0f32, -1.0f32];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("linear"), RString::from("2.0")));
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
            assert_eq!(out_slice, &[1.0, -1.0, 2.0, -2.0]);
        } else {
            panic!("Expected Payload::Audio");
        }
    }

    #[test]
    fn test_gain_db() {
        let input_samples = vec![1.0f32];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("db"), RString::from("6.0206")));
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
            assert!((out_slice[0] - 2.0).abs() < 1e-3);
        } else {
            panic!("Expected Payload::Audio");
        }
    }

    #[test]
    fn test_gain_audio_payload() {
        let input_samples = vec![0.5f32, -0.5f32, 1.0f32, -1.0f32];
        let audio = Audio::from_f32_planar(&input_samples, 2, 44100).unwrap();

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
