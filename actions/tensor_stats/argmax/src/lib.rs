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

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    let axis_arg = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()));
    if axis_arg.is_some_and(|value| value.starts_with('$')) {
        let keep = args.get_named("keepdim");
        if keep.is_some_and(|value| value.starts_with('$')) {
            return ShapeResult::Unknown;
        }
        let keep = keep.is_some_and(|value| value == "true" || value == "1");
        let rank = if keep {
            input.rank()
        } else if input.rank() == 0 {
            0
        } else {
            input.rank().saturating_sub(1).max(1)
        };
        return ShapeResult::Ok(Shape::unknown(rank));
    }
    // Validate the same selected argument that execution consumes.
    for value in [
        core_types::contract::value(&args, &["axis", "dim"], Some(0)),
        core_types::contract::value(&args, &["keepdim"], None),
    ] {
        if let Err(error) = value {
            return core_types::contract::finish(Err(error));
        }
    }
    if let Some(value) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if args.parse::<isize>(value).is_err() {
            return ShapeResult::Invalid(format!("Invalid axis argument '{value}'").into());
        }
    }

    let r = input.rank();
    let mut axis: isize = -1;
    let mut keepdim = false;
    if let Some(ax_str) = args
        .get_named("axis")
        .or_else(|| args.get_named("dim"))
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(ax) = args.parse::<isize>(ax_str) {
            axis = ax;
        }
    }
    if let Some(kd_str) = args.get_named("keepdim") {
        keepdim = kd_str == "true" || kd_str == "1";
    }

    if r == 0 {
        return ShapeResult::Ok(Shape::scalar());
    }
    let resolved = if axis < 0 { axis + r as isize } else { axis };
    if !(0..r as isize).contains(&resolved) {
        return ShapeResult::Invalid(core_types::reducer_axis_reason(axis, r).into());
    }
    let ax = resolved as usize;
    if input.dims()[ax] == 0 {
        return ShapeResult::Invalid(
            "Cannot reduce this operation along an empty dimension".into(),
        );
    }

    let mut out: Vec<core_types::Dimension> = Vec::new();
    for (i, &d) in input.dims().iter().enumerate() {
        if i == ax {
            if keepdim {
                out.push(1.into());
            }
        } else {
            out.push(d);
        }
    }
    if out.is_empty() {
        out.push(1.into());
    }
    ShapeResult::Ok(Shape::new(out))
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let inner_payload = payload.into_unwrapped();

    let axis = prepared.unsigned("axis").map(|value| value as usize);

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match reduce_argmax(
                &tensor,
                axis.unwrap_or(0),
                prepared
                    .output_dims()
                    .expect("shapecheck prepared dimensions"),
            ) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error("Action \'argmax\' requires a tensor or scalar value".into()),
    }
}

fn reduce_argmax(tensor: &Tensor, ax: usize, out_shape: Vec<usize>) -> Result<Tensor, String> {
    let vals = tensor.to_vec_f32();
    let r = tensor.rank();
    if r == 0 {
        return Ok(Tensor::from_i32_vec(vec![0], vec![]).unwrap());
    }

    let outer_size: usize = tensor.shape[0..ax].iter().product();
    let axis_len = tensor.shape[ax];

    let inner_size: usize = tensor.shape[(ax + 1)..r].iter().product();

    let out_len = outer_size * inner_size;
    if out_len == 0 {
        return Tensor::from_i32_vec(Vec::new(), out_shape).map_err(|e| e.to_string());
    }
    let mut out_indices = vec![0i32; out_len];

    out_indices
        .par_chunks_mut(inner_size)
        .enumerate()
        .for_each(|(outer_idx, slice)| {
            for (inner_idx, slot) in slice.iter_mut().enumerate() {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0i32;

                for a in 0..axis_len {
                    let in_idx = (outer_idx * axis_len + a) * inner_size + inner_idx;
                    let val = vals[in_idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = a as i32;
                    }
                }
                *slot = max_idx;
            }
        });

    Tensor::from_i32_vec(out_indices, out_shape).map_err(|e| e.to_string())
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
    core_types::shapecheck::axis_plan(result, rank, 0, Some(-1), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_argmax_action() {
        let tensor =
            Tensor::from_f32_shape(&[1.0, 20.0, 5.0, 100.0, 8.0, 9.0], vec![2, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_i32_slice().unwrap(), &[1, 0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}

#[test]
fn test_argmax_accepts_a_scalar_value() {
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
            assert_eq!(out.as_i32_slice().unwrap(), &[0]);
        }
        other => panic!(
            "scalar path produced the wrong payload: {}",
            core_types::payload_kind_name(&other)
        ),
    }
}
