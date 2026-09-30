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

fn shape_impl(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
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
    let out_len = ((input.len() as f32) * ratio).round() as usize;
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

    let mut from_rate = audio.sample_rate as f32;
    let mut to_rate = 44100.0f32;

    if let Some(args) = &args_opt {
        if let Some(fr) = args
            .get_named("from_rate")
            .or_else(|| args.get_named("source_rate"))
        {
            if let Ok(v) = fr.parse::<f32>() {
                from_rate = v;
            }
        }
        if let Some(tr) = args
            .get_named("to_rate")
            .or_else(|| args.get_named("rate"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(v) = tr.parse::<f32>() {
                to_rate = v;
            }
        }
    }

    let samples = match audio.as_f32_slice() {
        Some(s) => s,
        None => return Payload::Audio(audio),
    };
    let num_channels = audio.channels();
    let channel_len = audio.num_samples();

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

        let out_tensor = Tensor::from_f32_vec(combined, vec![num_channels, out_channel_len])
            .unwrap_or_else(|_| audio.tensor.clone());
        let out_audio = core_types::Audio {
            tensor: out_tensor,
            sample_rate: to_rate as u32,
            channel_layout: audio.channel_layout,
            layout: audio.layout,
        };
        Payload::Audio(out_audio)
    } else {
        let resampled = resample_channel_sinc(samples, from_rate, to_rate, 8);
        let out_len = resampled.len();
        let out_tensor =
            Tensor::from_f32_vec(resampled, vec![out_len]).unwrap_or_else(|_| audio.tensor.clone());
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
}
