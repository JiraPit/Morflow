use core_types::{DataType, Payload, Tensor, TensorDType};
use rayon::prelude::*;
use std::f32::consts::PI;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
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
    let mut from_rate = 48000.0f32;
    let mut to_rate = 44100.0f32;

    if let Payload::Audio(audio) = payload.unwrap_payload() {
        from_rate = audio.sample_rate as f32;
    }

    if let Some(args) = payload.args() {
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

    match payload.unwrap_payload() {
        Payload::Audio(audio) if audio.dtype() == TensorDType::F32 => {
            let bytes = audio.tensor.to_contiguous_bytes();
            let samples: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            let shape = audio.tensor.shape.as_slice();
            if shape.len() == 2 {
                let num_channels = shape[0];
                let channel_len = shape[1];

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

                let out_tensor =
                    Tensor::from_f32_shape(&combined, vec![num_channels, out_channel_len])
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
                let out_tensor = Tensor::from_f32_shape(&resampled, vec![out_len])
                    .unwrap_or_else(|_| audio.tensor.clone());
                let out_audio = core_types::Audio {
                    tensor: out_tensor,
                    sample_rate: to_rate as u32,
                    channel_layout: audio.channel_layout,
                    layout: audio.layout,
                };
                Payload::Audio(out_audio)
            }
        }
        Payload::Tensor(tensor) if tensor.dtype == TensorDType::F32 => {
            let bytes = tensor.to_contiguous_bytes();
            let samples: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            let shape = tensor.shape.as_slice();
            if shape.len() == 2 {
                let num_channels = shape[0];
                let channel_len = shape[1];

                let channel_slices: Vec<&[f32]> = (0..num_channels)
                    .map(|ch| &samples[ch * channel_len..(ch + 1) * channel_len])
                    .collect();

                // Resample all channels in parallel across Rayon workers
                let resampled_channels: Vec<Vec<f32>> = channel_slices
                    .into_par_iter()
                    .map(|ch| resample_channel_sinc(ch, from_rate, to_rate, 8))
                    .collect();

                let out_channel_len = resampled_channels.first().map(|v| v.len()).unwrap_or(0);
                let mut combined = Vec::with_capacity(num_channels * out_channel_len);
                for ch_data in resampled_channels {
                    combined.extend_from_slice(&ch_data);
                }

                let out_tensor =
                    Tensor::from_f32_shape(&combined, vec![num_channels, out_channel_len])
                        .unwrap_or_else(|_| tensor.clone());
                Payload::Tensor(out_tensor)
            } else {
                let resampled = resample_channel_sinc(samples, from_rate, to_rate, 8);
                let out_len = resampled.len();
                let out_tensor = Tensor::from_f32_shape(&resampled, vec![out_len])
                    .unwrap_or_else(|_| tensor.clone());
                Payload::Tensor(out_tensor)
            }
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_resample_ratio() {
        // 100 samples at 48000 Hz resampled to 24000 Hz -> 50 samples
        let input_samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 100]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("from_rate"), RString::from("48000.0")));
        named.push(Tuple2(RString::from("to_rate"), RString::from("24000.0")));
        let args = ActionArgs {
            positional: core_types::RVec::new(),
            named,
        };

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args,
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[1, 50]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
