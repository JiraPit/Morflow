use core_types::{DataType, Payload, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut width = 1.2f32; // 0.0 = mono, 1.0 = unchanged, >1.0 = wider
    let mut center_gain_db = 0.0f32;

    if let Some(args) = payload.args() {
        if let Some(w) = args.get_named("width").or_else(|| args.get_named("amount")) {
            if let Ok(v) = w.parse::<f32>() {
                width = v.max(0.0);
            }
        }
        if let Some(cg) = args
            .get_named("center_gain_db")
            .or_else(|| args.get_named("center"))
        {
            if let Ok(v) = cg.parse::<f32>() {
                center_gain_db = v;
            }
        }
    }

    let mid_gain = 10.0f32.powf(center_gain_db / 20.0);
    let inv_sqrt2 = 1.0f32 / std::f32::consts::SQRT_2;

    match payload.unwrap_payload() {
        Payload::Audio(audio) if audio.dtype() == TensorDType::F32 => {
            let shape = audio.tensor.shape.as_slice();
            // Expecting [2, samples] or [channels, samples] with channels >= 2
            if shape.len() == 2 && shape[0] >= 2 {
                let num_samples = shape[1];
                let mut bytes = audio.tensor.to_contiguous_bytes();
                let all_samples: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr() as *mut f32,
                        bytes.len() / std::mem::size_of::<f32>(),
                    )
                };

                let (left_channel, rest) = all_samples.split_at_mut(num_samples);
                let (right_channel, _) = rest.split_at_mut(num_samples);

                // Process Mid/Side matrix in parallel across sample chunks
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

                let out_tensor = Tensor::from_f32_shape(all_samples, audio.tensor.shape.to_vec())
                    .unwrap_or_else(|_| audio.tensor.clone());
                let out_audio = core_types::Audio {
                    tensor: out_tensor,
                    sample_rate: audio.sample_rate,
                    channel_layout: audio.channel_layout,
                    layout: audio.layout,
                };
                Payload::Audio(out_audio)
            } else {
                Payload::Audio(audio.clone())
            }
        }
        Payload::Tensor(tensor) if tensor.dtype == TensorDType::F32 => {
            let shape = tensor.shape.as_slice();
            // Expecting [2, samples] or [channels, samples] with channels >= 2
            if shape.len() == 2 && shape[0] >= 2 {
                let num_samples = shape[1];
                let mut bytes = tensor.to_contiguous_bytes();
                let all_samples: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr() as *mut f32,
                        bytes.len() / std::mem::size_of::<f32>(),
                    )
                };

                let (left_channel, rest) = all_samples.split_at_mut(num_samples);
                let (right_channel, _) = rest.split_at_mut(num_samples);

                // Process Mid/Side matrix in parallel across sample chunks
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

                let out_tensor = Tensor::from_f32_shape(all_samples, tensor.shape.to_vec())
                    .unwrap_or_else(|_| tensor.clone());
                Payload::Tensor(out_tensor)
            } else {
                // If mono or 1D tensor, widening is a no-op
                Payload::Tensor(tensor.clone())
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
    fn test_stereo_widen_mono_collapse() {
        // [left: 1.0, right: 0.0] with width = 0.0 (collapse to mono)
        let input_samples = vec![1.0f32, 0.0f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![2, 1]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("width"), RString::from("0.0")));
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
            let out_slice: &[f32] = out_t.as_f32_slice().unwrap();
            // Both channels should be equal to 0.5 (mid component distributed equally)
            assert!((out_slice[0] - 0.5).abs() < 1e-4);
            assert!((out_slice[1] - 0.5).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
