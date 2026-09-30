use core_types::{ActionArgs, DataType, GetShapeFn, Payload, Shape, ShapeResult};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: ActionArgs) -> ShapeResult {
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
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(Error::from("Flattened dimension overflows"))?;
        out.push(flat);
        out.extend_from_slice(&input.dims()[end + 1..]);
        contract::shape(out)
    })())
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut start_dim = 0usize;
    let mut end_dim = -1isize;

    if let Some(args) = &args_opt {
        if let Some(s_str) = args
            .get_named("start_dim")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(s) = s_str.parse::<usize>() {
                start_dim = s;
            }
        }
        if let Some(e_str) = args
            .get_named("end_dim")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(e) = e_str.parse::<isize>() {
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

#[cfg(test)]
mod tests {
    use super::*;
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
        let res = process(Payload::scalar_f32(2.0));
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
