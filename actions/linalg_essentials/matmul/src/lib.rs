use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite | DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: core_types::ValueShape,
    args: A,
) -> core_types::ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::matmul(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, _) = (payload.into_unwrapped(), Some(&prepared.args));

    match inner_payload {
        Payload::Composite(items) if items.len() == 2 => {
            let (t1, t2) = match (&items[0], &items[1]) {
                (Payload::Tensor(a), Payload::Tensor(b)) => (a, b),
                _ => return Payload::Error("matmul expects 2 tensor inputs in composite".into()),
            };
            match compute_matmul(t1, t2, &prepared) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => {
            // Self-multiplication A * A
            match compute_matmul(&t, &t, &prepared) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        other => other,
    }
}

fn compute_matmul(
    a: &Tensor,
    b: &Tensor,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    let (r_a, r_b) = (a.rank(), b.rank());
    let out_shape = prepared.output_dims().map_err(|e| e.to_string())?;
    let output_len = out_shape.iter().product::<usize>();
    if output_len == 0 {
        return Tensor::from_f32_vec(Vec::new(), out_shape).map_err(|e| e.to_string());
    }
    let (m, k_a, n) = (a.shape[r_a - 2], a.shape[r_a - 1], b.shape[r_b - 1]);
    let output_batch = &out_shape[..out_shape.len() - 2];
    let batch_index = |mut index: usize, input_batch: &[usize]| {
        let mut flat = 0;
        let mut stride = 1;
        for axis in (0..output_batch.len()).rev() {
            let coordinate = index % output_batch[axis];
            index /= output_batch[axis];
            if axis + input_batch.len() >= output_batch.len() {
                let dim = input_batch[axis + input_batch.len() - output_batch.len()];
                if dim != 1 {
                    flat += coordinate * stride;
                }
                stride *= dim;
            }
        }
        flat
    };
    let a_vals = a.to_vec_f32();
    let b_vals = b.to_vec_f32();

    let mat_a_size = m * k_a;
    let mat_b_size = k_a * n;
    let mat_c_size = m * n;

    let mut out_vals = vec![0.0f32; output_len];

    out_vals
        .par_chunks_mut(mat_c_size)
        .enumerate()
        .for_each(|(batch_idx, c_mat)| {
            let a_offset = batch_index(batch_idx, &a.shape[..r_a - 2]) * mat_a_size;
            let b_offset = batch_index(batch_idx, &b.shape[..r_b - 2]) * mat_b_size;

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

    Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn get_action_abi_version() -> u32 {
    core_types::shapecheck::ACTION_ABI_VERSION
}
#[no_mangle]
pub extern "C" fn get_action_abi_layout() -> *const core_types::abi_stable::type_layout::TypeLayout
{
    <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT
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
        None,
        Some(get_output_value_shape),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
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
