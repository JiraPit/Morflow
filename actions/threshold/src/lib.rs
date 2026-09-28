use core_types::{DataType, Image, Payload, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ThreshMode {
    Binary,
    BinaryInv,
    Otsu,
    Truncate,
    ToZero,
    ToZeroInv,
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut threshold_opt: Option<f32> = None;
    let mut max_val_opt: Option<f32> = None;
    let mut mode = ThreshMode::Binary;

    if let Some(args) = payload.args() {
        if let Some(t_str) = args
            .get_named("threshold")
            .or_else(|| args.get_named("thresh"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            threshold_opt = t_str.parse::<f32>().ok();
        }
        if let Some(m_str) = args.get_named("max_val").or_else(|| args.get_named("max")) {
            max_val_opt = m_str.parse::<f32>().ok();
        }
        if let Some(mode_str) = args.get_named("mode") {
            mode = match mode_str.to_lowercase().as_str() {
                "binary_inv" | "inv" => ThreshMode::BinaryInv,
                "otsu" => ThreshMode::Otsu,
                "truncate" | "trunc" => ThreshMode::Truncate,
                "to_zero" | "tozero" => ThreshMode::ToZero,
                "to_zero_inv" => ThreshMode::ToZeroInv,
                _ => ThreshMode::Binary,
            };
        }
    }

    match payload.unwrap_payload() {
        Payload::Image(img) => {
            let res = apply_threshold(&img.tensor, threshold_opt, max_val_opt, mode);
            Payload::Image(Image {
                tensor: res,
                color_space: img.color_space,
                layout: img.layout,
            })
        }
        Payload::Tensor(tensor) => {
            let res = apply_threshold(tensor, threshold_opt, max_val_opt, mode);
            Payload::Tensor(res)
        }
        other => other.clone(),
    }
}

fn apply_threshold(
    tensor: &Tensor,
    thresh_opt: Option<f32>,
    max_opt: Option<f32>,
    mode: ThreshMode,
) -> Tensor {
    match tensor.dtype {
        TensorDType::F32 => {
            let mut bytes = tensor.to_contiguous_bytes();
            let slice: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4)
            };

            let max_val = max_opt.unwrap_or(1.0);
            let threshold = if mode == ThreshMode::Otsu {
                compute_otsu_threshold_f32(slice)
            } else {
                thresh_opt.unwrap_or(0.5)
            };

            slice.par_iter_mut().for_each(|val| {
                *val = match mode {
                    ThreshMode::Binary | ThreshMode::Otsu => {
                        if *val >= threshold {
                            max_val
                        } else {
                            0.0
                        }
                    }
                    ThreshMode::BinaryInv => {
                        if *val >= threshold {
                            0.0
                        } else {
                            max_val
                        }
                    }
                    ThreshMode::Truncate => {
                        if *val > threshold {
                            threshold
                        } else {
                            *val
                        }
                    }
                    ThreshMode::ToZero => {
                        if *val >= threshold {
                            *val
                        } else {
                            0.0
                        }
                    }
                    ThreshMode::ToZeroInv => {
                        if *val >= threshold {
                            0.0
                        } else {
                            *val
                        }
                    }
                };
            });

            Tensor::from_f32_shape(slice, tensor.shape.to_vec()).unwrap()
        }
        TensorDType::U8 => {
            let mut bytes = tensor.to_contiguous_bytes();
            let slice: &mut [u8] = bytes.as_mut_slice();

            let max_val = max_opt
                .map(|m| m.clamp(0.0, 255.0).round() as u8)
                .unwrap_or(255);
            let threshold = if mode == ThreshMode::Otsu {
                compute_otsu_threshold_u8(slice)
            } else {
                thresh_opt
                    .map(|t| t.clamp(0.0, 255.0).round() as u8)
                    .unwrap_or(128)
            };

            slice.par_iter_mut().for_each(|val| {
                *val = match mode {
                    ThreshMode::Binary | ThreshMode::Otsu => {
                        if *val >= threshold {
                            max_val
                        } else {
                            0
                        }
                    }
                    ThreshMode::BinaryInv => {
                        if *val >= threshold {
                            0
                        } else {
                            max_val
                        }
                    }
                    ThreshMode::Truncate => {
                        if *val > threshold {
                            threshold
                        } else {
                            *val
                        }
                    }
                    ThreshMode::ToZero => {
                        if *val >= threshold {
                            *val
                        } else {
                            0
                        }
                    }
                    ThreshMode::ToZeroInv => {
                        if *val >= threshold {
                            0
                        } else {
                            *val
                        }
                    }
                };
            });

            Tensor::from_rvec_u8(bytes, tensor.shape.to_vec(), TensorDType::U8).unwrap()
        }
        _ => tensor.clone(),
    }
}

fn compute_otsu_threshold_u8(slice: &[u8]) -> u8 {
    let mut hist = [0usize; 256];
    for &val in slice {
        hist[val as usize] += 1;
    }

    let total = slice.len() as f64;
    if total == 0.0 {
        return 128;
    }

    let mut sum_total = 0.0f64;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut weight_bg = 0.0f64;
    let mut sum_bg = 0.0f64;
    let mut max_variance = 0.0f64;
    let mut best_threshold = 0u8;

    for (t, &count) in hist.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += t as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;

        let var_between = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if var_between > max_variance {
            max_variance = var_between;
            best_threshold = t as u8;
        }
    }

    best_threshold
}

fn compute_otsu_threshold_f32(slice: &[f32]) -> f32 {
    let mut u8_samples = vec![0u8; slice.len()];
    u8_samples
        .par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(dst, &src)| {
            *dst = (src.clamp(0.0, 1.0) * 255.0).round() as u8;
        });

    let best_t = compute_otsu_threshold_u8(&u8_samples);
    best_t as f32 / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_binary_threshold() {
        let f32_data = vec![0.1f32, 0.4, 0.6, 0.9];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![2, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("threshold"), RString::from("0.5")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            let slice: &[f32] = out_t.as_f32_slice().unwrap();
            assert_eq!(slice, &[0.0, 0.0, 1.0, 1.0]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
