use core_types::{DataType, Payload, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Composite(items) if items.len() >= 2 => {
            let (t1, t2) = match (&items[0], &items[1]) {
                (Payload::Tensor(a), Payload::Tensor(b)) => (a, b),
                _ => return Payload::Error("matmul expects 2 tensor inputs in composite".into()),
            };
            match compute_matmul(t1, t2) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => {
            // Self-multiplication A * A
            match compute_matmul(&t, &t) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        other => other,
    }
}

fn compute_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let r_a = a.rank();
    let r_b = b.rank();

    if r_a < 2 || r_b < 2 {
        return Err(format!(
            "matmul requires tensors of rank at least 2, got {} and {}",
            r_a, r_b
        ));
    }

    let m = a.shape[r_a - 2];
    let k_a = a.shape[r_a - 1];
    let k_b = b.shape[r_b - 2];
    let n = b.shape[r_b - 1];

    if k_a != k_b {
        return Err(format!(
            "Matrix inner dimensions mismatch: {} vs {}",
            k_a, k_b
        ));
    }

    let batch_a: usize = a.shape[0..r_a - 2].iter().product();
    let batch_b: usize = b.shape[0..r_b - 2].iter().product();

    let num_batches = batch_a.max(batch_b);
    let a_vals = a.to_vec_f32();
    let b_vals = b.to_vec_f32();

    let mat_a_size = m * k_a;
    let mat_b_size = k_a * n;
    let mat_c_size = m * n;

    let mut out_vals = vec![0.0f32; num_batches * mat_c_size];

    out_vals
        .par_chunks_mut(mat_c_size)
        .enumerate()
        .for_each(|(batch_idx, c_mat)| {
            let a_offset = (batch_idx % batch_a) * mat_a_size;
            let b_offset = (batch_idx % batch_b) * mat_b_size;

            let cur_a = &a_vals[a_offset..a_offset + mat_a_size];
            let cur_b = &b_vals[b_offset..b_offset + mat_b_size];

            for i in 0..m {
                let a_row = &cur_a[i * k_a..(i + 1) * k_a];
                let c_row = &mut c_mat[i * n..(i + 1) * n];
                for p in 0..k_a {
                    let a_ip = a_row[p];
                    let b_row = &cur_b[p * n..(p + 1) * n];
                    for j in 0..n {
                        c_row[j] += a_ip * b_row[j];
                    }
                }
            }
        });

    let mut out_shape = if r_a >= r_b {
        a.shape[0..r_a - 2].to_vec()
    } else {
        b.shape[0..r_b - 2].to_vec()
    };
    out_shape.push(m);
    out_shape.push(n);

    Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{RVec, Tensor};

    #[test]
    fn test_matmul_action() {
        // [2, 3] x [3, 2] -> [2, 2]
        let a = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_f32_shape(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0], vec![3, 2]).unwrap();

        let mut items = RVec::new();
        items.push(Payload::Tensor(a));
        items.push(Payload::Tensor(b));

        let res = process(Payload::Composite(items));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            // [1*7+2*9+3*2, 1*8+2*1+3*3] = [7+18+6, 8+2+9] = [31, 19]
            // [4*7+5*9+6*2, 4*8+5*1+6*3] = [28+45+12, 32+5+18] = [85, 55]
            assert_eq!(out.as_f32_slice().unwrap(), &[31.0, 19.0, 85.0, 55.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
