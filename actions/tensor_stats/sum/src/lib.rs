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
    let mut axis: Option<isize> = None;
    let mut keepdim = false;
    if let Some(ax_str) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(ax) = ax_str.parse::<isize>() {
            axis = Some(ax);
        }
    }
    if let Some(kd_str) = args.get_named("keepdim") {
        keepdim = kd_str == "true" || kd_str == "1";
    }

    let mut out: Vec<usize> = Vec::new();
    match axis {
        Some(raw) => {
            let ax = if raw < 0 {
                ((raw + r as isize).max(0)) as usize
            } else {
                raw as usize
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
        }
        None => {
            if keepdim {
                out = vec![1; r.max(1)];
            }
        }
    }
    Shape::new(out)
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut axis = None;
    let mut keepdim = false;

    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = ax_str.parse::<isize>() {
                axis = Some(ax);
            }
        }
        if let Some(kd_str) = args.get_named("keepdim") {
            keepdim = kd_str == "true" || kd_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_sum(&tensor, axis, keepdim) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'sum\' requires a tensor or scalar value".into()),
    }
}

fn reduce_sum(tensor: &Tensor, axis: Option<isize>, keepdim: bool) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();

    if let Some(raw_ax) = axis {
        let ax = if raw_ax < 0 {
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

        if ax >= r {
            return Err(format!(
                "Axis {} out of bounds for tensor of rank {}",
                raw_ax, r
            ));
        }

        let outer_size: usize = tensor.shape[0..ax].iter().product();
        let axis_len = tensor.shape[ax];
        let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

        let out_len = outer_size * inner_size;
        let mut out_vals = vec![0.0f32; out_len];

        out_vals
            .par_chunks_mut(inner_size)
            .enumerate()
            .for_each(|(outer_idx, slice)| {
                for (inner_idx, slot) in slice.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for a in 0..axis_len {
                        let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                        acc += vals[in_idx];
                    }
                    *slot = acc;
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

        Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
    } else {
        let total: f32 = vals.par_iter().sum();
        let shape = if keepdim {
            vec![1; r.max(1)]
        } else {
            Vec::new()
        };
        Tensor::from_f32_vec(vec![total], shape).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_sum_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Sum along axis 0
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("0")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor.clone())),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[4.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }

        // Sum along axis -1
        let mut named_neg = core_types::RVec::new();
        named_neg.push(Tuple2(RString::from("axis"), RString::from("-1")));
        let payload_neg = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor.clone())),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named: named_neg,
            },
        };

        let res_neg = process(payload_neg);
        if let Payload::Tensor(out) = res_neg {
            assert_eq!(out.shape.as_slice(), &[2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[3.0, 7.0]);
        } else {
            panic!("Expected Tensor output");
        }

        // Total sum
        let res_tot = process(Payload::Tensor(tensor));
        if let Payload::Scalar(out) = res_tot {
            assert_eq!(out.as_f32_slice().unwrap(), &[10.0]);
        } else {
            panic!("Expected Scalar output");
        }
    }
}

#[test]
fn test_sum_accepts_a_scalar_value() {
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
