use core_types::{ActionArgs, DataType, GetShapeFn, Payload, RString, Shape, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
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

    let samples = audio.tensor.as_f32_slice_mut();
    samples.par_iter_mut().for_each(|s| *s *= multiplier);
    Payload::Audio(audio)
}

#[cfg(test)]
mod tests {
    use super::*;
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
