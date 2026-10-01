use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg, axis};
    contract::finish((|| {
        let dim = match arg::<isize>(&args, &["axis", "dim"], Some(0), Some(-1)) {
            Err(core_types::contract::Error::Unknown) => return Ok(input),
            other => other?.unwrap(),
        };
        if input.rank() > 0 {
            let dim = if dim < 0 {
                (input.rank() as isize).saturating_add(dim).max(0)
            } else {
                dim
            };
            axis(dim, input.rank(), false)?;
        }
        Ok(input)
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
    let inner_payload = payload.into_unwrapped();
    let axis = prepared.unsigned("axis").unwrap_or(0) as usize;

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match compute_log_softmax(&tensor, axis) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'log_softmax\' requires a tensor or scalar value",
        )),
    }
}

fn compute_log_softmax(tensor: &Tensor, ax: usize) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(Tensor::from_f32_vec(vec![0.0], vec![]).unwrap());
    }

    let axis_len = tensor.shape[ax];
    if axis_len == 0 || vals.is_empty() {
        return Ok(tensor.clone());
    }
    let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

    let mut out_vals = vec![0.0f32; vals.len()];

    out_vals
        .par_chunks_mut(axis_len * inner_size)
        .enumerate()
        .for_each(|(outer_idx, block)| {
            for inner_idx in 0..inner_size {
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    max_val = max_val.max(vals[in_idx]);
                }

                let mut sum_exp = 0.0f32;
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    sum_exp += (vals[in_idx] - max_val).exp();
                }

                let log_sum_exp = max_val + sum_exp.ln();
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    block[a * inner_size + inner_idx] = vals[in_idx] - log_sum_exp;
                }
            }
        });

    Tensor::from_f32_vec(out_vals, tensor.shape.to_vec()).map_err(|e| e.to_string())
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
    let rank = input.value.shape().map(Shape::rank);
    let result = core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    );
    core_types::shapecheck::axis_plan(result, rank, 0, Some(-1), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_log_softmax_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            let exp_sum: f32 = slice.iter().map(|&x| x.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
