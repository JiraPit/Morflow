use core_types::{DataType, Payload, TensorDType};
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
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
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

    match inner_payload {
        Payload::Audio(mut audio) if audio.dtype() == TensorDType::F32 => {
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
        Payload::Tensor(mut tensor) if tensor.dtype == TensorDType::F32 => {
            let current_level = if mode == "rms" {
                tensor.rms() as f32
            } else {
                tensor.peak_abs() as f32
            };

            if current_level < 1e-8 {
                return Payload::Tensor(tensor);
            }

            let scale = target_peak / current_level;
            let samples = tensor.as_f32_slice_mut();
            samples.par_iter_mut().for_each(|s| *s *= scale);
            Payload::Tensor(tensor)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_normalize_peak() {
        let input_samples = vec![0.2f32, -0.5f32, 0.1f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("target_peak"), RString::from("1.0")));
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
            assert!((out_t.peak_abs() - 1.0).abs() < 1e-5);
            assert_eq!(out_slice, &[0.4, -1.0, 0.2]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }

    #[test]
    fn test_normalize_target_db() {
        let input_samples = vec![0.5f32, -0.5f32];
        let tensor = Tensor::from_f32_shape(&input_samples, vec![1, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("target_db"), RString::from("0.0")));
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
            assert!((out_t.peak_abs() - 1.0).abs() < 1e-5);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
