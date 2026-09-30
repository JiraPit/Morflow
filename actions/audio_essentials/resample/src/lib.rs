use core_types::{
    ActionArgs, DataType, GetShapeFn, Payload, RString, Shape, ShapeResult, Tensor, TensorDType,
};
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

fn rate(value: Option<&str>, fallback: Option<f32>, label: &str) -> Result<Option<f32>, RString> {
    let value = match value {
        Some(value) if value.starts_with('$') => return Ok(None),
        Some(value) => Some(value.parse::<f32>().map_err(|_| {
            RString::from(format!("resample {label} must be a positive finite rate"))
        })?),
        None => fallback,
    };
    if value.is_some_and(|v| !v.is_finite() || v <= 0.0) {
        return Err(format!("resample {label} must be a positive finite rate").into());
    }
    Ok(value)
}

fn rates(args: &ActionArgs, source: Option<f32>) -> Result<(Option<f32>, Option<f32>), RString> {
    let from = rate(
        args.get_named("from_rate")
            .or_else(|| args.get_named("source_rate")),
        source,
        "from_rate",
    )?;
    let to = rate(
        args.get_named("to_rate")
            .or_else(|| args.get_named("rate"))
            .or_else(|| args.positional.first().map(|s| s.as_str())),
        Some(44100.0),
        "to_rate",
    )?;
    Ok((from, to))
}

fn output_len(samples: usize, from: f32, to: f32) -> Result<usize, RString> {
    if samples == 0 || (from - to).abs() < 0.1 {
        return Ok(samples);
    }
    let length = (samples as f32 * (to / from)).round();
    if !length.is_finite() || length >= usize::MAX as f32 {
        return Err("resample output sample count overflows".into());
    }
    Ok(length as usize)
}

fn shape_impl(input: Shape, args: ActionArgs) -> ShapeResult {
    if !matches!(input.rank(), 1 | 2) {
        return ShapeResult::Invalid(
            "resample requires mono [samples] or planar [channels, samples] audio".into(),
        );
    }
    let (from, to) = match rates(&args, None) {
        Ok(rates) => rates,
        Err(error) => return ShapeResult::Invalid(error),
    };
    let (Some(from), Some(to)) = (from, to) else {
        return ShapeResult::Unknown;
    };
    let mut dims = input.dims().to_vec();
    if dims.len() == 2 {
        if dims[0] == 0 {
            // Unknown channels could be mono, whose runtime output is rank 1.
            return ShapeResult::Unknown;
        }
        if dims[0] == 1 {
            dims.remove(0);
        }
    }
    let last = dims.len() - 1;
    if dims[last] != 0 {
        dims[last] = match output_len(dims[last], from, to) {
            Ok(length) => length,
            Err(error) => return ShapeResult::Invalid(error),
        };
    }
    core_types::contract::finish(core_types::contract::shape(dims))
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
}

/// Computes windowed sinc interpolation for high-quality audio resampling.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

fn resample_channel_sinc(
    input: &[f32],
    from_rate: f32,
    to_rate: f32,
    filter_half_len: isize,
) -> Vec<f32> {
    if input.is_empty() || (from_rate - to_rate).abs() < 0.1 {
        return input.to_vec();
    }

    let ratio = to_rate / from_rate;
    // Rates and the maximum output length are validated by process.
    let out_len = output_len(input.len(), from_rate, to_rate).expect("validated output length");
    let mut output = Vec::with_capacity(out_len);

    let cutoff = if ratio < 1.0 { ratio * 0.95 } else { 0.95 };

    for i in 0..out_len {
        let center_in = (i as f32) / ratio;
        let left_idx = (center_in.floor() as isize) - filter_half_len;
        let right_idx = (center_in.ceil() as isize) + filter_half_len;

        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for j in left_idx..=right_idx {
            if j >= 0 && (j as usize) < input.len() {
                let dt = (j as f32) - center_in;
                // Blackman-Harris window
                let norm_t = dt / (filter_half_len as f32);
                if norm_t.abs() <= 1.0 {
                    let w = 0.5 * (1.0 + (PI * norm_t).cos());
                    let sinc_val = sinc(cutoff * dt) * cutoff;
                    let weight = w * sinc_val;
                    sum += input[j as usize] * weight;
                    weight_sum += weight;
                }
            }
        }

        if weight_sum.abs() > 1e-6 {
            output.push(sum / weight_sum);
        } else {
            output.push(0.0);
        }
    }

    output
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'resample' requires Payload::Audio"));
        }
    };

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'resample' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let args = args_opt.unwrap_or_default();
    let (from_rate, to_rate) = match rates(&args, Some(audio.sample_rate as f32)) {
        Ok((Some(from), Some(to))) => (from, to),
        Ok(_) => {
            return Payload::Error("resample sample rates must be resolved at execution".into())
        }
        Err(error) => return Payload::Error(error),
    };
    if !matches!(audio.tensor.rank(), 1 | 2)
        || (audio.tensor.rank() == 2 && audio.layout != core_types::AudioLayout::Planar)
        || audio.channels() == 0
    {
        return Payload::Error(
            "resample requires mono [samples] or planar [channels, samples] audio".into(),
        );
    }
    let num_channels = audio.channels();
    let channel_len = audio.num_samples();
    if let Err(error) = output_len(channel_len, from_rate, to_rate).and_then(|len| {
        len.checked_mul(num_channels)
            .ok_or_else(|| RString::from("resample output element count overflows"))
    }) {
        return Payload::Error(error);
    }
    let samples = match audio.as_f32_slice() {
        Some(samples) => std::borrow::Cow::Borrowed(samples),
        None => std::borrow::Cow::Owned(audio.to_vec_f32()),
    };

    if num_channels > 1 {
        let channel_slices: Vec<&[f32]> = (0..num_channels)
            .map(|ch| &samples[ch * channel_len..(ch + 1) * channel_len])
            .collect();

        let resampled_channels: Vec<Vec<f32>> = channel_slices
            .into_par_iter()
            .map(|ch| resample_channel_sinc(ch, from_rate, to_rate, 8))
            .collect();

        let out_channel_len = resampled_channels.first().map(|v| v.len()).unwrap_or(0);
        let mut combined = Vec::with_capacity(num_channels * out_channel_len);
        for ch_data in resampled_channels {
            combined.extend_from_slice(&ch_data);
        }

        let out_tensor = match Tensor::from_f32_vec(combined, vec![num_channels, out_channel_len]) {
            Ok(tensor) => tensor,
            Err(error) => return Payload::Error(error),
        };
        let out_audio = core_types::Audio {
            tensor: out_tensor,
            sample_rate: to_rate as u32,
            channel_layout: audio.channel_layout,
            layout: audio.layout,
        };
        Payload::Audio(out_audio)
    } else {
        let resampled = resample_channel_sinc(&samples, from_rate, to_rate, 8);
        let out_len = resampled.len();
        let out_tensor = match Tensor::from_f32_vec(resampled, vec![out_len]) {
            Ok(tensor) => tensor,
            Err(error) => return Payload::Error(error),
        };
        let out_audio = core_types::Audio {
            tensor: out_tensor,
            sample_rate: to_rate as u32,
            channel_layout: audio.channel_layout,
            layout: audio.layout,
        };
        Payload::Audio(out_audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_resample_ratio() {
        // 100 samples at 48000 Hz resampled to 24000 Hz -> 50 samples
        let input_samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let audio = Audio::from_f32_planar(&input_samples, 1, 48000).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("from_rate"), RString::from("48000.0")));
        named.push(Tuple2(RString::from("to_rate"), RString::from("24000.0")));
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
            assert_eq!(out_aud.sample_rate, 24000);
            assert_eq!(out_aud.num_samples(), 50);
        } else {
            panic!("Expected Payload::Audio");
        }
    }
    fn assert_verdict(actual: ShapeResult, expected: ShapeResult) {
        match (actual, expected) {
            (ShapeResult::Ok(actual), ShapeResult::Ok(expected)) => assert_eq!(actual, expected),
            (ShapeResult::Unknown, ShapeResult::Unknown) => {}
            (actual, expected) => panic!("Expected {expected:?}, got {actual:?}"),
        }
    }

    fn args(from: Option<&str>, to: Option<&str>) -> ActionArgs {
        let mut args = ActionArgs::default();
        for (key, value) in [("from_rate", from), ("to_rate", to)] {
            if let Some(value) = value {
                args.named.push(Tuple2(key.into(), value.into()));
            }
        }
        args
    }

    #[test]
    fn mono_and_planar_shapes_match_runtime_rounding_and_identity_rates() {
        for channels in [1, 2] {
            for (from, to) in [
                (48000.0, 24000.0),
                (24000.0, 48000.0),
                (48000.0, 44100.0),
                (48000.0, 48000.05),
            ] {
                let audio =
                    Audio::from_f32_planar(&vec![0.0; channels * 101], channels, from as u32)
                        .unwrap();
                let args = args(Some(&from.to_string()), Some(&to.to_string()));
                let predicted =
                    get_output_shape(Shape::new(audio.tensor.shape.iter().copied()), args.clone());
                let result = process(Payload::WithArgs {
                    payload: RBox::new(Payload::Audio(audio)),
                    args,
                });
                match result {
                    Payload::Audio(audio) => assert_verdict(
                        predicted,
                        ShapeResult::Ok(Shape::new(audio.tensor.shape.iter().copied())),
                    ),
                    other => panic!("Unexpected {other:?}"),
                }
            }
        }
        // A planar one-channel input becomes the runtime's canonical mono shape.
        assert_verdict(
            get_output_shape(Shape::new([1, 101]), args(Some("48000"), Some("24000"))),
            ShapeResult::Ok(Shape::new([51])),
        );
    }

    #[test]
    fn unknown_metadata_and_dimensions_remain_unknown() {
        assert_verdict(
            get_output_shape(Shape::new([100]), args(None, Some("24000"))),
            ShapeResult::Unknown,
        );
        assert_verdict(
            get_output_shape(Shape::new([100]), args(Some("$rate"), None)),
            ShapeResult::Unknown,
        );
        assert_verdict(
            get_output_shape(Shape::new([0, 100]), args(Some("48000"), None)),
            ShapeResult::Unknown,
        );
        assert_verdict(
            get_output_shape(Shape::new([2, 0]), args(Some("48000"), Some("24000"))),
            ShapeResult::Ok(Shape::new([2, 0])),
        );
        let mut aliases = ActionArgs::default();
        aliases
            .named
            .push(Tuple2("source_rate".into(), "48000".into()));
        aliases.positional.push("24000".into());
        assert_verdict(
            get_output_shape(Shape::new([100]), aliases),
            ShapeResult::Ok(Shape::new([50])),
        );
    }

    #[test]
    fn rejects_invalid_rates_at_check_and_runtime() {
        for bad in ["0", "-1", "NaN", "inf", "oops"] {
            for args in [
                args(Some(bad), Some("24000")),
                args(Some("48000"), Some(bad)),
            ] {
                assert!(matches!(
                    get_output_shape(Shape::new([100]), args.clone()),
                    ShapeResult::Invalid(_)
                ));
                let audio = Audio::from_f32_planar(&[0.0; 100], 1, 48000).unwrap();
                assert!(matches!(
                    process(Payload::WithArgs {
                        payload: RBox::new(Payload::Audio(audio)),
                        args
                    }),
                    Payload::Error(_)
                ));
            }
        }
        assert!(matches!(
            get_output_shape(Shape::new([2, 3, 4]), args(Some("48000"), None)),
            ShapeResult::Invalid(_)
        ));
    }
    #[test]
    fn source_metadata_and_layout_are_handled_at_execution() {
        let mut audio = Audio::from_f32_planar(&[0.0; 101], 1, 48000).unwrap();
        audio.tensor = audio.tensor.reshape(vec![1, 101]).unwrap();
        let metadata_args = args(None, Some("24000"));
        assert!(matches!(
            get_output_shape(Shape::new([1, 101]), metadata_args.clone()),
            ShapeResult::Unknown
        ));
        match process(Payload::WithArgs {
            payload: RBox::new(Payload::Audio(audio)),
            args: metadata_args,
        }) {
            Payload::Audio(audio) => assert_eq!(audio.tensor.shape.as_slice(), &[51]),
            other => panic!("Unexpected {other:?}"),
        }
        let audio = Audio::from_f32_interleaved(&[0.0; 200], 2, 48000).unwrap();
        assert!(matches!(
            process(Payload::WithArgs {
                payload: RBox::new(Payload::Audio(audio)),
                args: args(Some("48000"), Some("24000"))
            }),
            Payload::Error(_)
        ));
    }
}
