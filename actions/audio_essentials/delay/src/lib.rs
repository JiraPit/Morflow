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
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut time_ms = 120.0f32;
    let mut feedback = 0.35f32;
    let mut mix = 0.3f32;
    let mut sample_rate = 44100.0f32;

    if let Payload::Audio(audio) = payload.unwrap_payload() {
        sample_rate = audio.sample_rate as f32;
    }

    if let Some(args) = payload.args() {
        if let Some(t) = args
            .get_named("time_ms")
            .or_else(|| args.get_named("time"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(v) = t.parse::<f32>() {
                time_ms = v;
            }
        }
        if let Some(fb) = args.get_named("feedback") {
            if let Ok(v) = fb.parse::<f32>() {
                feedback = v;
            }
        }
        if let Some(m) = args.get_named("mix").or_else(|| args.get_named("wet")) {
            if let Ok(v) = m.parse::<f32>() {
                mix = v;
            }
        }
        if let Some(sr) = args
            .get_named("sample_rate")
            .or_else(|| args.get_named("rate"))
        {
            if let Ok(v) = sr.parse::<f32>() {
                sample_rate = v;
            }
        }
    }

    let params = DelayParams::new(time_ms, feedback, mix, sample_rate);

    match payload.unwrap_payload() {
        Payload::Audio(audio) if audio.dtype() == TensorDType::F32 => {
            let mut bytes = audio.tensor.to_contiguous_bytes();
            let samples: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            let shape = audio.tensor.shape.as_slice();
            if shape.len() == 2 {
                let channel_len = shape[1];
                if channel_len > 0 {
                    samples
                        .par_chunks_mut(channel_len)
                        .for_each(|ch| params.process_channel(ch));
                }
            } else {
                params.process_channel(samples);
            }

            let out_tensor = Tensor::from_f32_shape(samples, audio.tensor.shape.to_vec())
                .unwrap_or_else(|_| audio.tensor.clone());
            let out_audio = core_types::Audio {
                tensor: out_tensor,
                sample_rate: audio.sample_rate,
                channel_layout: audio.channel_layout,
                layout: audio.layout,
            };
            Payload::Audio(out_audio)
        }
        Payload::Tensor(tensor) if tensor.dtype == TensorDType::F32 => {
            let mut bytes = tensor.to_contiguous_bytes();
            let samples: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_mut_ptr() as *mut f32,
                    bytes.len() / std::mem::size_of::<f32>(),
                )
            };

            let shape = tensor.shape.as_slice();
            if shape.len() == 2 {
                let channel_len = shape[1];
                if channel_len > 0 {
                    samples
                        .par_chunks_mut(channel_len)
                        .for_each(|ch| params.process_channel(ch));
                }
            } else {
                params.process_channel(samples);
            }

            let out_tensor = Tensor::from_f32_shape(samples, tensor.shape.to_vec())
                .unwrap_or_else(|_| tensor.clone());
            Payload::Tensor(out_tensor)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_delay_echo() {
        // Single impulse at index 0 [1.0, 0.0, 0.0, 0.0]
        let input_samples = vec![1.0f32, 0.0f32, 0.0f32, 0.0f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 4]).unwrap();

        // Delay 2 samples at 1000 Hz = 2 ms
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("time_ms"), RString::from("2.0")));
        named.push(Tuple2(RString::from("feedback"), RString::from("0.0")));
        named.push(Tuple2(RString::from("mix"), RString::from("0.5")));
        named.push(Tuple2(
            RString::from("sample_rate"),
            RString::from("1000.0"),
        ));
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
            // Dry at index 0: 0.5 * 1.0 = 0.5
            assert!((out_slice[0] - 0.5).abs() < 1e-4);
            // Echo at index 2 (2 samples later): 0.5 * 1.0 = 0.5
            assert!((out_slice[2] - 0.5).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
