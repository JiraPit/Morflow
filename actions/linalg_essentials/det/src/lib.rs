use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if input.rank() < 2 {
            return Err("det requires rank at least 2".into());
        }
        let rank = input.rank();
        let (h, w) = (input.dims()[rank - 2], input.dims()[rank - 1]);
        if !h.is_unknown() && !w.is_unknown() && h != w {
            return Err("Matrix must be square".into());
        }
        contract::shape(input.dims()[..rank - 2].iter().copied())
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, _) = (payload.into_unwrapped(), Some(&prepared.args));

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_det(&tensor) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_det(tensor: &Tensor) -> Result<Tensor, String> {
    let r = tensor.rank();

    let n = tensor.shape[r - 2];

    let batch_size: usize = tensor.shape[0..r - 2].iter().product();
    let mat_size = n * n;
    let vals = tensor.to_vec_f32();

    let mut out_dets = vec![0.0f32; batch_size];

    out_dets
        .par_iter_mut()
        .enumerate()
        .for_each(|(b, det_out)| {
            let offset = b * mat_size;
            let mut mat = vals[offset..offset + mat_size].to_vec();

            let mut det = 1.0f32;
            let mut sign = 1.0f32;

            for col in 0..n {
                // Pivot selection
                let mut pivot_row = col;
                let mut max_pivot = mat[col * n + col].abs();
                for row in (col + 1)..n {
                    let v = mat[row * n + col].abs();
                    if v > max_pivot {
                        max_pivot = v;
                        pivot_row = row;
                    }
                }

                if max_pivot < 1e-12 {
                    det = 0.0;
                    break;
                }

                if pivot_row != col {
                    for j in 0..n {
                        mat.swap(col * n + j, pivot_row * n + j);
                    }
                    sign = -sign;
                }

                let pivot_val = mat[col * n + col];
                det *= pivot_val;

                for row in (col + 1)..n {
                    let factor = mat[row * n + col] / pivot_val;
                    for j in (col + 1)..n {
                        mat[row * n + j] -= factor * mat[col * n + j];
                    }
                }
            }

            *det_out = det * sign;
        });

    let out_shape = if r > 2 {
        tensor.shape[0..r - 2].to_vec()
    } else {
        Vec::new()
    };

    Tensor::from_f32_vec(out_dets, out_shape).map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_det_action() {
        // [4, 7; 2, 6] -> det = 24 - 14 = 10
        let mat = Tensor::from_f32_shape(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Scalar(out) = res {
            assert!((out.as_f32_slice().unwrap()[0] - 10.0).abs() < 1e-4);
        } else {
            panic!("Expected Scalar output");
        }
    }
    #[test]
    fn test_det_returns_scalar_for_tensor_input() {
        let tensor = Tensor::from_f32_shape(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(tensor));
        match res {
            Payload::Scalar(out) => {
                assert!(out.as_f32_slice().is_some());
            }
            other => panic!(
                "action did not return a scalar: {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }
}
