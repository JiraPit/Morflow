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
                _ => return Payload::Error("outer expects 2 tensor inputs in composite".into()),
            };
            match compute_outer(t1, t2) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => match compute_outer(&t, &t) {
            Ok(out) => Payload::Tensor(out),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_outer(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let vals_a = a.to_vec_f32();
    let vals_b = b.to_vec_f32();

    let m = vals_a.len();
    let n = vals_b.len();

    let mut out_vals = vec![0.0f32; m * n];

    out_vals.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_i = vals_a[i];
        for j in 0..n {
            row[j] = a_i * vals_b[j];
        }
    });

    Tensor::from_f32_vec(out_vals, vec![m, n]).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{RVec, Tensor};

    #[test]
    fn test_outer_action() {
        let t1 = Tensor::from_f32_slice(&[1.0, 2.0]);
        let t2 = Tensor::from_f32_slice(&[3.0, 4.0, 5.0]);

        let mut items = RVec::new();
        items.push(Payload::Tensor(t1));
        items.push(Payload::Tensor(t2));

        let res = process(Payload::Composite(items));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 3]);
            assert_eq!(
                out.as_f32_slice().unwrap(),
                &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            );
        } else {
            panic!("Expected Tensor output");
        }
    }
}
