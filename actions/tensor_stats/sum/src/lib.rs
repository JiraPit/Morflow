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
        if let Some(kd_str) = args.get_named("keepdim") {
            keepdim = kd_str == "true" || kd_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match reduce_sum(&tensor, axis, keepdim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Image(image) => match reduce_sum(&image.tensor, axis, keepdim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Audio(audio) => match reduce_sum(&audio.tensor, axis, keepdim) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn reduce_sum(tensor: &Tensor, axis: Option<usize>, keepdim: bool) -> Result<Tensor, String> {
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
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
    } else {
        let total: f32 = vals.par_iter().sum();
        let shape = if keepdim { vec![1; r.max(1)] } else { vec![1] };
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

        // Total sum
        let res_tot = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res_tot {
            assert_eq!(out.as_f32_slice().unwrap(), &[10.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
