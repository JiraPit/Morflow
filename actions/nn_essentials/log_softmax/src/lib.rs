use core_types::{ActionArgs, DataType, GetShapeResultFn, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeResultFn = get_output_shape_result;

#[no_mangle]
pub extern "C" fn get_output_shape_result(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut axis = -1isize;
    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<isize>() {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match compute_log_softmax(&tensor, axis) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'log_softmax\' requires a tensor or scalar value",
        )),
    }
}

fn compute_log_softmax(tensor: &Tensor, axis_raw: isize) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(Tensor::from_f32_vec(vec![0.0], vec![]).unwrap());
    }

    let ax = if axis_raw < 0 {
        (r as isize + axis_raw).max(0) as usize
    } else {
        axis_raw as usize
    };

    if ax >= r {
        return Err(format!(
            "Axis {} out of bounds for tensor of rank {}",
            ax, r
        ));
    }

    let axis_len = tensor.shape[ax];
    if axis_len == 0 {
        return Ok(tensor.clone());
    }
    let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

    let mut out_vals = vec![0.0f32; vals.len()];

    out_vals
        .par_chunks_mut(axis_len * inner_size)
        .enumerate()
        .for_each(|(outer_idx, block)| {
            for inner_idx in 0..inner_size {
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    max_val = max_val.max(vals[in_idx]);
                }

                let mut sum_exp = 0.0f32;
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    sum_exp += (vals[in_idx] - max_val).exp();
                }

                let log_sum_exp = max_val + sum_exp.ln();
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    block[a * inner_size + inner_idx] = vals[in_idx] - log_sum_exp;
                }
            }
        });

    Tensor::from_f32_vec(out_vals, tensor.shape.to_vec()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_log_softmax_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            let exp_sum: f32 = slice.iter().map(|&x| x.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
