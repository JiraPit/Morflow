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

    let mut axis = 0isize;

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
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_cumsum(&tensor, axis) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error("Action \'cumsum\' requires a tensor or scalar value".into()),
    }
}

fn compute_cumsum(tensor: &Tensor, raw_ax: isize) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(tensor.clone());
    }

    let axis = if raw_ax < 0 {
        let pos = raw_ax + r as isize;
        if pos < 0 {
            return Err(format!(
                "Axis {} out of bounds for tensor of rank {}",
                raw_ax, r
            ));
        }
        pos as usize
    } else {
        raw_ax as usize
    };

    if axis >= r {
        return Err(format!(
            "Axis {} out of bounds for tensor of rank {}",
            raw_ax, r
        ));
    }

    let axis_len = tensor.shape[axis];
    let inner_size: usize = tensor.shape[(axis + 1)..r].iter().product();

    let mut out_vals = vec![0.0f32; vals.len()];

    out_vals
        .par_chunks_mut(axis_len * inner_size)
        .enumerate()
        .for_each(|(outer_idx, block)| {
            for inner_idx in 0..inner_size {
                let mut acc = 0.0f32;
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    acc += vals[in_idx];
                    block[a * inner_size + inner_idx] = acc;
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
    fn test_cumsum_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[test]
fn test_cumsum_accepts_a_scalar_value() {
    let res = process(Payload::scalar_f32(2.0));
    match res {
        Payload::Scalar(out) => {
            assert_eq!(out.as_f32_slice().unwrap(), &[2.0]);
        }
        other => panic!(
            "scalar path produced the wrong payload: {}",
            core_types::payload_kind_name(&other)
        ),
    }
}
