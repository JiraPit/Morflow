use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult, Tensor};
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
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub static MORFLOW_SHAPE_ABI: u32 = core_types::SHAPE_ABI_VERSION;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
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
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_softmax(&tensor, axis) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error(core_types::RString::from(
            "Action \'softmax\' requires a tensor or scalar value",
        )),
    }
}

fn compute_softmax(tensor: &Tensor, axis_raw: isize) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(Tensor::from_f32_vec(vec![1.0], vec![]).unwrap());
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
                    let exp_val = (vals[in_idx] - max_val).exp();
                    block[a * inner_size + inner_idx] = exp_val;
                    sum_exp += exp_val;
                }

                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
                for a in 0..axis_len {
                    block[a * inner_size + inner_idx] *= inv_sum;
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
    fn test_softmax_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
            assert!(slice[2] > slice[1] && slice[1] > slice[0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
