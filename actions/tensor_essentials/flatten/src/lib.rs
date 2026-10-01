use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg, Error};
    contract::finish((|| {
        let rank = input.rank();
        let start = arg::<usize>(&args, &["start_dim"], Some(0), Some(0))?.unwrap();
        let end = arg::<isize>(&args, &["end_dim"], Some(1), Some(-1))?.unwrap();
        if rank == 0 {
            return Ok(input);
        }
        let end = if end < 0 {
            (rank as isize).saturating_add(end).max(0) as usize
        } else {
            (end as usize).min(rank - 1)
        };
        if start >= rank || start > end {
            return Err("Invalid flatten dimension range".into());
        }
        let mut out = input.dims()[..start].to_vec();
        let flat = input.dims()[start..=end]
            .iter()
            .try_fold(core_types::Dimension::Known(1), |n, d| n.checked_mul(*d))
            .ok_or(Error::from("Flattened dimension overflows"))?;
        out.push(flat);
        out.extend_from_slice(&input.dims()[end + 1..]);
        contract::shape(out)
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
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut start_dim = 0usize;
    let mut end_dim = -1isize;

    if let Some(args) = &args_opt {
        if let Some(s_str) = args
            .get_named("start_dim")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(s) = prepared.args.parse::<usize>(s_str) {
                start_dim = s;
            }
        }
        if let Some(e_str) = args
            .get_named("end_dim")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(e) = prepared.args.parse::<isize>(e_str) {
                end_dim = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match tensor.flatten(start_dim, end_dim) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'flatten\' requires a tensor or scalar value",
        )),
    }
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
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_flatten_action() {
        let tensor =
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
                .unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("start_dim"), RString::from("1")));
        named.push(Tuple2(RString::from("end_dim"), RString::from("-1")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 4]);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_flatten_accepts_a_scalar_value() {
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
}
