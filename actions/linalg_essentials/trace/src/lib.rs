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
    let (inner_payload, _) = payload.take_payload_and_args();

    match inner_payload {
        Payload::Tensor(tensor) => match compute_trace(&tensor) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error("Action 'trace' requires Payload::Tensor".into()),
    }
}

fn compute_trace(tensor: &Tensor) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r < 2 {
        return Err("trace requires matrix of rank at least 2".into());
    }

    let h = tensor.shape[r - 2];
    let w = tensor.shape[r - 1];
    let diag_len = h.min(w);

    let batch_size: usize = tensor.shape[0..r - 2].iter().product();
    let mat_size = h * w;
    let vals = tensor.to_vec_f32();

    let mut out_traces = vec![0.0f32; batch_size];

    out_traces.par_iter_mut().enumerate().for_each(|(b, tr)| {
        let offset = b * mat_size;
        let mut sum = 0.0f32;
        for i in 0..diag_len {
            sum += vals[offset + i * w + i];
        }
        *tr = sum;
    });

    let out_shape = if r > 2 {
        tensor.shape[0..r - 2].to_vec()
    } else {
        vec![1]
    };

    Tensor::from_f32_vec(out_traces, out_shape).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::Tensor;

    #[test]
    fn test_trace_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[5.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
