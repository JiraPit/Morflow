use core_types::{ActionArgs, DataType, GetShapeResultFn, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
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
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_inv(&tensor) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_inv(tensor: &Tensor) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r < 2 {
        return Err("inv requires tensor of rank at least 2".into());
    }

    let n = tensor.shape[r - 2];
    let m = tensor.shape[r - 1];
    if n != m {
        return Err(format!(
            "Matrix must be square for inversion, got {}x{}",
            n, m
        ));
    }

    let batch_size: usize = tensor.shape[0..r - 2].iter().product();
    let mat_size = n * n;
    let vals = tensor.to_vec_f32();

    let mut out_vals = vec![0.0f32; batch_size * mat_size];

    let results: Result<Vec<()>, String> = out_vals
        .par_chunks_mut(mat_size)
        .enumerate()
        .map(|(b, inv_mat)| {
            let offset = b * mat_size;
            let src = &vals[offset..offset + mat_size];

            // Augmented matrix [A | I] of size n x 2n
            let mut aug = vec![0.0f32; n * 2 * n];
            for i in 0..n {
                for j in 0..n {
                    aug[i * 2 * n + j] = src[i * n + j];
                }
                aug[i * 2 * n + n + i] = 1.0;
            }

            // Gauss-Jordan elimination with partial pivoting
            for col in 0..n {
                // Find pivot row
                let mut pivot_row = col;
                let mut max_pivot = aug[col * 2 * n + col].abs();
                for row in (col + 1)..n {
                    let val = aug[row * 2 * n + col].abs();
                    if val > max_pivot {
                        max_pivot = val;
                        pivot_row = row;
                    }
                }

                if max_pivot < 1e-12 {
                    return Err(format!(
                        "Matrix in batch {} is singular or non-invertible",
                        b
                    ));
                }

                // Swap rows if necessary
                if pivot_row != col {
                    for j in 0..2 * n {
                        aug.swap(col * 2 * n + j, pivot_row * 2 * n + j);
                    }
                }

                // Normalize pivot row
                let pivot_val = aug[col * 2 * n + col];
                let inv_pivot = 1.0 / pivot_val;
                for j in 0..2 * n {
                    aug[col * 2 * n + j] *= inv_pivot;
                }

                // Eliminate other rows
                for row in 0..n {
                    if row != col {
                        let factor = aug[row * 2 * n + col];
                        if factor.abs() > 1e-12 {
                            for j in 0..2 * n {
                                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                            }
                        }
                    }
                }
            }

            // Copy inverse from right side of augmented matrix
            for i in 0..n {
                for j in 0..n {
                    inv_mat[i * n + j] = aug[i * 2 * n + n + j];
                }
            }

            Ok(())
        })
        .collect();

    results?;

    Tensor::from_f32_vec(out_vals, tensor.shape.to_vec()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_inv_action() {
        // [4, 7; 2, 6] -> det = 24 - 14 = 10 -> inv = [0.6, -0.7; -0.2, 0.4]
        let mat = Tensor::from_f32_shape(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!((slice[0] - 0.6).abs() < 1e-4);
            assert!((slice[1] - (-0.7)).abs() < 1e-4);
            assert!((slice[2] - (-0.2)).abs() < 1e-4);
            assert!((slice[3] - 0.4).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
