use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> Shape {
    let r = input.rank();
    let mut axis: isize = -1;
    let mut keepdim = false;
    if let Some(ax_str) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(ax) = ax_str.parse::<isize>() {
            axis = ax;
        }
    }
    if let Some(kd_str) = args.get_named("keepdim") {
        keepdim = kd_str == "true" || kd_str == "1";
    }

    let mut out: Vec<usize> = Vec::new();
    if r == 0 {
        return Shape::new(out);
    }
    let ax = if axis < 0 {
        ((axis + r as isize).max(0)) as usize
    } else {
        axis as usize
    };
    if ax >= r {
        return input;
    }
    for (i, &d) in input.dims().iter().enumerate() {
        if i == ax {
            if keepdim {
                out.push(1);
            }
        } else {
            out.push(d);
        }
    }
    if out.is_empty() {
        out.push(1);
    }
    Shape::new(out)
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut axis = -1isize;
    let mut keepdim = false;

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
        if let Some(kd_str) = args.get_named("keepdim") {
            keepdim = kd_str == "true" || kd_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_argmin(&tensor, axis, keepdim) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'argmin\' requires a tensor or scalar value".into()),
    }
}

fn reduce_argmin(tensor: &Tensor, axis_raw: isize, keepdim: bool) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(Tensor::from_i32_vec(vec![0], vec![]).unwrap());
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

    let outer_size: usize = tensor.shape[0..ax].iter().product();
    let axis_len = tensor.shape[ax];
    if axis_len == 0 {
        return Err("Cannot compute argmin along empty dimension".into());
    }
    let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

    let out_len = outer_size * inner_size;
    let mut out_indices = vec![0i32; out_len];

    out_indices
        .par_chunks_mut(inner_size)
        .enumerate()
        .for_each(|(outer_idx, slice)| {
            for (inner_idx, slot) in slice.iter_mut().enumerate() {
                let mut min_val = f32::INFINITY;
                let mut min_idx = 0i32;

                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    let val = vals[in_idx];
                    if val < min_val {
                        min_val = val;
                        min_idx = a as i32;
                    }
                }
                *slot = min_idx;
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

    Tensor::from_i32_vec(out_indices, out_shape).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_argmin_action() {
        let tensor = Tensor::from_f32_shape(&[10.0, 2.0, 5.0, 1.0, 8.0, 9.0], vec![2, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_i32_slice().unwrap(), &[1, 0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[test]
fn test_argmin_accepts_a_scalar_value() {
    let res = process(Payload::scalar_f32(2.0));
    match res {
        Payload::Scalar(out) => {
            assert_eq!(out.as_i32_slice().unwrap(), &[0]);
        }
        other => panic!(
            "scalar path produced the wrong payload: {}",
            core_types::payload_kind_name(&other)
        ),
    }
}
