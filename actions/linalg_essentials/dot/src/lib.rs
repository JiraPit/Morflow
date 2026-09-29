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
                _ => return Payload::Error("dot expects 2 tensor inputs in composite".into()),
            };
            match compute_dot(t1, t2) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => match compute_dot(&t, &t) {
            Ok(out) => Payload::Tensor(out),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_dot(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let vals_a = a.to_vec_f32();
    let vals_b = b.to_vec_f32();

    if vals_a.len() != vals_b.len() {
        return Err(format!(
            "Vector length mismatch in dot product: {} vs {}",
            vals_a.len(),
            vals_b.len()
        ));
    }

    let dot: f32 = vals_a
        .par_iter()
        .zip(vals_b.par_iter())
        .map(|(&x, &y)| x * y)
        .sum();

    Ok(Tensor::from_f32_slice(&[dot]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{RVec, Tensor};

    #[test]
    fn test_dot_action() {
        let t1 = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);
        let t2 = Tensor::from_f32_slice(&[4.0, 5.0, 6.0]);

        let mut items = RVec::new();
        items.push(Payload::Tensor(t1));
        items.push(Payload::Tensor(t2));

        let res = process(Payload::Composite(items));
        if let Payload::Tensor(out) = res {
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert_eq!(out.as_f32_slice().unwrap(), &[32.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
