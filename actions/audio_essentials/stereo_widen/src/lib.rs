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
            return Payload::Error(RString::from(
                "Action 'stereo_widen' requires Payload::Audio",
            ));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'stereo_widen' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut width = 1.2f32; // 0.0 = mono, 1.0 = unchanged, >1.0 = wider
    let mut center_gain_db = 0.0f32;

    if let Some(args) = &args_opt {
        if let Some(w) = args.get_named("width").or_else(|| args.get_named("amount")) {
            if let Ok(v) = prepared.args.parse::<f32>(w) {
                width = v.max(0.0);
            }
        }
        if let Some(cg) = args
            .get_named("center_gain_db")
            .or_else(|| args.get_named("center"))
        {
            if let Ok(v) = prepared.args.parse::<f32>(cg) {
                center_gain_db = v;
            }
        }
    }

    let mid_gain = 10.0f32.powf(center_gain_db / 20.0);
    let inv_sqrt2 = 1.0f32 / std::f32::consts::SQRT_2;

    if audio.channels() >= 2 {
        let num_samples = audio.num_samples();
        let all_samples = audio.tensor.as_f32_slice_mut();
        let (left_channel, rest) = all_samples.split_at_mut(num_samples);
        let (right_channel, _) = rest.split_at_mut(num_samples);

        left_channel
            .par_iter_mut()
            .zip(right_channel.par_iter_mut())
            .for_each(|(l, r)| {
                let left = *l;
                let right = *r;

                let mid = (left + right) * inv_sqrt2 * mid_gain;
                let side = (left - right) * inv_sqrt2 * width;

                *l = (mid + side) * inv_sqrt2;
                *r = (mid - side) * inv_sqrt2;
            });
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
    fn test_stereo_widen_mono_collapse() {
        // [left: 1.0, right: 0.0] with width = 0.0 (collapse to mono)
        let input_samples = vec![1.0f32, 0.0f32];
        let audio = Audio::from_f32_planar(&input_samples, 2, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("width"), RString::from("0.0")));
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
            // Both channels should be equal to 0.5 (mid component distributed equally)
            assert!((out_slice[0] - 0.5).abs() < 1e-4);
            assert!((out_slice[1] - 0.5).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Audio");
        }
    }
}
