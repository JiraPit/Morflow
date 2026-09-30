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

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'normalize' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'normalize' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    // 1. Resolve normalization parameters
    let mut target_peak = 1.0f32;
    let mut mode = "peak";

    if let Some(args) = &args_opt {
        if let Some(m) = args.get_named("mode") {
            mode = m;
        }
        if let Some(p_str) = args.get_named("target_peak") {
            if let Ok(p) = p_str.parse::<f32>() {
                target_peak = p;
            }
        }
        if let Some(db_str) = args
            .get_named("target_peak_db")
            .or_else(|| args.get_named("target_db"))
        {
            if let Ok(db) = db_str.parse::<f32>() {
                target_peak = 10.0f32.powf(db / 20.0);
            }
        }
    }

    let current_level = if mode == "rms" {
        audio.tensor.rms() as f32
    } else {
        audio.tensor.peak_abs() as f32
    };

    if current_level < 1e-8 {
        return Payload::Audio(audio);
    }

    let scale = target_peak / current_level;
    let samples = audio.tensor.as_f32_slice_mut();
    samples.par_iter_mut().for_each(|s| *s *= scale);
    Payload::Audio(audio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_normalize_peak() {
        let input_samples = vec![0.2f32, -0.5f32, 0.1f32];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("target_peak"), RString::from("1.0")));
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
            assert!((out_aud.tensor.peak_abs() - 1.0).abs() < 1e-5);
            assert_eq!(out_slice, &[0.4, -1.0, 0.2]);
        } else {
            panic!("Expected Payload::Audio");
        }
    }

    #[test]
    fn test_normalize_target_db() {
        let input_samples = vec![0.5f32, -0.5f32];
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("target_db"), RString::from("0.0")));
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
            assert!((out_aud.tensor.peak_abs() - 1.0).abs() < 1e-5);
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
