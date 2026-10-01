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
            if let Ok(p) = prepared.args.parse::<f32>(p_str) {
                target_peak = p;
            }
        }
        if let Some(db_str) = args
            .get_named("target_peak_db")
            .or_else(|| args.get_named("target_db"))
        {
            if let Ok(db) = prepared.args.parse::<f32>(db_str) {
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
