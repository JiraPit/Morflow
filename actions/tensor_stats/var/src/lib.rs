use core_types::{DataType, Payload, Tensor};
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
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut axis = None;
    let mut unbiased = true;
    let mut keepdim = false;

    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<usize>() {
                axis = Some(ax);
            }
        }
        if let Some(unb_str) = args.get_named("unbiased") {
            unbiased = unb_str != "false" && unb_str != "0";
        }
        if let Some(kd_str) = args.get_named("keepdim") {
            keepdim = kd_str == "true" || kd_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match reduce_var(&tensor, axis, unbiased, keepdim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Image(image) => match reduce_var(&image.tensor, axis, unbiased, keepdim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Audio(audio) => match reduce_var(&audio.tensor, axis, unbiased, keepdim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn reduce_var(
    tensor: &Tensor,
    axis: Option<usize>,
    unbiased: bool,
    keepdim: bool,
) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();

    if let Some(ax) = axis {
        if ax >= r {
            return Err(format!(
                "Axis {} out of bounds for tensor of rank {}",
                ax, r
            ));
        }

        let outer_size: usize = tensor.shape[0..ax].iter().product();
        let axis_len = tensor.shape[ax];
        if axis_len == 0 {
            return Err("Cannot compute variance along empty dimension".into());
        }
        let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

        let out_len = outer_size * inner_size;
        let mut out_vals = vec![0.0f32; out_len];
        let divisor = if unbiased && axis_len > 1 {
            (axis_len - 1) as f32
        } else {
            axis_len as f32
        };

        out_vals
            .par_chunks_mut(inner_size)
            .enumerate()
            .for_each(|(outer_idx, slice)| {
                for (inner_idx, slot) in slice.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for a in 0..axis_len {
                        let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                        sum += vals[in_idx];
                    }
                    let mean = sum / axis_len as f32;

                    let mut sum_sq_diff = 0.0f32;
                    for a in 0..axis_len {
                        let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                        let diff = vals[in_idx] - mean;
                        sum_sq_diff += diff * diff;
                    }
                    *slot = sum_sq_diff / divisor;
                }
            });

        let mut out_shape = Vec::new();
        for (i, &dim) in tensor.shape.iter().enumerate() {
            if i == ax {
                if keepdim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(dim);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
    } else {
        let n = vals.len();
        if n == 0 {
            return Ok(Tensor::from_f32_slice(&[0.0]));
        }
        let sum: f32 = vals.par_iter().sum();
        let mean = sum / n as f32;
        let sum_sq_diff: f32 = vals.par_iter().map(|&x| (x - mean) * (x - mean)).sum();
        let divisor = if unbiased && n > 1 {
            (n - 1) as f32
        } else {
            n as f32
        };
        let var = sum_sq_diff / divisor;

        let shape = if keepdim { vec![1; r.max(1)] } else { vec![1] };
        Tensor::from_f32_vec(vec![var], shape).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_var_action() {
        let tensor = Tensor::from_f32_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            // sample variance = 4.5714
            assert!((slice[0] - 4.5714).abs() < 1e-3);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
