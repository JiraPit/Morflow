use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult, Tensor};
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
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_cholesky(&tensor) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_cholesky(a: &Tensor) -> Result<Tensor, String> {
    let rank = a.rank();
    if rank != 2 {
        return Err("cholesky requires 2D square matrix [N, N]".into());
    }

    let n = a.shape[0];
    let m = a.shape[1];
    if n != m {
        return Err(format!(
            "Matrix must be square for cholesky, got {}x{}",
            n, m
        ));
    }

    let a_vals = a.to_vec_f32();
    let mut l_vals = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l_vals[i * n + k] * l_vals[j * n + k];
            }

            if i == j {
                let diag = a_vals[i * n + i] - sum;
                if diag <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive-definite at index ({}, {})",
                        i, i
                    ));
                }
                l_vals[i * n + j] = diag.sqrt();
            } else {
                let l_jj = l_vals[j * n + j];
                if l_jj == 0.0 {
                    return Err(format!("Division by zero in Cholesky at ({}, {})", i, j));
                }
                l_vals[i * n + j] = (a_vals[i * n + j] - sum) / l_jj;
            }
        }
    }

    Tensor::from_f32_vec(l_vals, vec![n, n]).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_cholesky_action() {
        // A = [4, 12, -16; 12, 37, -43; -16, -43, 98]
        // L = [2, 0, 0; 6, 1, 0; -8, 5, 3]
        let a = Tensor::from_f32_shape(
            &[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
            vec![3, 3],
        )
        .unwrap();

        let res = process(Payload::Tensor(a));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[3, 3]);
            let slice = out.as_f32_slice().unwrap();
            assert_eq!(slice, &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0,]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
