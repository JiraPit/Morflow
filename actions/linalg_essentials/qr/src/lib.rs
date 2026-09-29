use core_types::{DataType, Payload, RVec, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Composite
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(tensor) => match compute_qr(&tensor) {
            Ok((q, r)) => {
                let mut items = RVec::new();
                items.push(Payload::Tensor(q));
                items.push(Payload::Tensor(r));
                Payload::Composite(items)
            }
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_qr(a: &Tensor) -> Result<(Tensor, Tensor), String> {
    let rank = a.rank();
    if rank != 2 {
        return Err("qr decomposition requires 2D matrix [M, N]".into());
    }

    let m = a.shape[0];
    let n = a.shape[1];
    let a_vals = a.to_vec_f32();

    // Modified Gram-Schmidt Orthogonalization
    let mut q_vals = vec![0.0f32; m * n];
    let mut r_vals = vec![0.0f32; n * n];

    // Initialize Q with columns of A
    for j in 0..n {
        for i in 0..m {
            q_vals[i * n + j] = a_vals[i * n + j];
        }
    }

    for k in 0..n {
        // Compute norm of column k in Q
        let mut norm_sq = 0.0f32;
        for i in 0..m {
            let v = q_vals[i * n + k];
            norm_sq += v * v;
        }
        let r_kk = norm_sq.sqrt();
        r_vals[k * n + k] = r_kk;

        if r_kk > 1e-12 {
            let inv_r_kk = 1.0 / r_kk;
            for i in 0..m {
                q_vals[i * n + k] *= inv_r_kk;
            }
        }

        // Project column k onto remaining columns
        for j in (k + 1)..n {
            let mut dot = 0.0f32;
            for i in 0..m {
                dot += q_vals[i * n + k] * q_vals[i * n + j];
            }
            r_vals[k * n + j] = dot;

            for i in 0..m {
                q_vals[i * n + j] -= dot * q_vals[i * n + k];
            }
        }
    }

    let q = Tensor::from_f32_vec(q_vals, vec![m, n]).map_err(|e| e.to_string())?;
    let r_mat = Tensor::from_f32_vec(r_vals, vec![n, n]).map_err(|e| e.to_string())?;

    Ok((q, r_mat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_qr_action() {
        let mat = Tensor::from_f32_shape(
            &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            vec![3, 3],
        )
        .unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Composite(items) = res {
            assert_eq!(items.len(), 2);
            if let (Payload::Tensor(q), Payload::Tensor(r)) = (&items[0], &items[1]) {
                assert_eq!(q.shape.as_slice(), &[3, 3]);
                assert_eq!(r.shape.as_slice(), &[3, 3]);
            } else {
                panic!("Expected Tensor Q and R");
            }
        } else {
            panic!("Expected Composite output");
        }
    }
}
