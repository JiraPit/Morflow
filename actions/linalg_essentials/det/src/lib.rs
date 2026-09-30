use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_shape(_input: Shape, _args: ActionArgs) -> Shape {
    Shape::new(Vec::new())
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, _) = payload.take_payload_and_args();

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
    if r < 2 {
        return Err("det requires tensor of rank at least 2".into());
    }

    let n = tensor.shape[r - 2];
    let m = tensor.shape[r - 1];
    if n != m {
        return Err(format!(
            "Matrix must be square for determinant, got {}x{}",
            n, m
        ));
    }

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

#[cfg(test)]
mod tests {
    use super::*;
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
