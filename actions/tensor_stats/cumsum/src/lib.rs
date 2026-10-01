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
    let rank = input.rank();
    let result = contract::finish((|| {
        let dim = arg::<isize>(&args, &["axis", "dim"], Some(0), Some(0))?.unwrap();
        if input.rank() > 0 {
            axis(dim, input.rank(), false)?;
        }
        Ok(input)
    })());
    match result {
        ShapeResult::Unknown => ShapeResult::Ok(Shape::unknown(rank)),
        other => other,
    }
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
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut axis = 0isize;

    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = prepared.args.parse::<isize>(ax_str) {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_cumsum(&tensor, axis) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error("Action \'cumsum\' requires a tensor or scalar value".into()),
    }
}

fn compute_cumsum(tensor: &Tensor, raw_ax: isize) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(tensor.clone());
    }

    let axis = if raw_ax < 0 {
        let pos = raw_ax + r as isize;
        if pos < 0 {
            return Err(format!(
                "Axis {} out of bounds for tensor of rank {}",
                raw_ax, r
            ));
        }
        pos as usize
    } else {
        raw_ax as usize
    };

    if axis >= r {
        return Err(format!(
            "Axis {} out of bounds for tensor of rank {}",
            raw_ax, r
        ));
    }

    let axis_len = tensor.shape[axis];
    let inner_size: usize = tensor.shape[(axis + 1)..r].iter().product();

    if vals.is_empty() {
        return Ok(tensor.clone());
    }
    let mut out_vals = vec![0.0f32; vals.len()];

    out_vals
        .par_chunks_mut(axis_len * inner_size)
        .enumerate()
        .for_each(|(outer_idx, block)| {
            for inner_idx in 0..inner_size {
                let mut acc = 0.0f32;
                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    acc += vals[in_idx];
                    block[a * inner_size + inner_idx] = acc;
                }
            }
        });

    Tensor::from_f32_vec(out_vals, tensor.shape.to_vec()).map_err(|e| e.to_string())
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
        Some(get_output_shape),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_cumsum_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[test]
fn test_cumsum_accepts_a_scalar_value() {
    let payload = Payload::scalar_f32(2.0);
    let core_types::ShapeCheckResult::Ready { prepared, .. } = shapecheck(
        core_types::InputDescriptor::from_payload(&payload),
        core_types::ActionArgs::default(),
    ) else {
        panic!("expected ready")
    };
    let res = crate::process(payload, prepared);
    match res {
        Payload::Scalar(out) => {
            assert_eq!(out.as_f32_slice().unwrap(), &[2.0]);
        }
        other => panic!(
            "scalar path produced the wrong payload: {}",
            core_types::payload_kind_name(&other)
        ),
    }
}
