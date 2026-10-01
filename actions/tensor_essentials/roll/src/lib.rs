use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
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
        if let Err(error @ core_types::contract::Error::Invalid(_)) =
            arg::<isize>(&args, &["shift", "shifts"], Some(0), Some(0))
        {
            return Err(error);
        }
        let dim = match arg::<isize>(&args, &["axis", "dim"], Some(1), Some(0)) {
            Err(core_types::contract::Error::Unknown) => return Ok(input),
            other => other?.unwrap(),
        };
        if input.rank() > 0 {
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
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut shift = 0isize;
    let axis = prepared.unsigned("axis").unwrap_or(0) as usize;

    if let Some(args) = &args_opt {
        if let Some(sh_str) = args
            .get_named("shift")
            .or_else(|| args.get_named("shifts"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(sh) = prepared.args.parse::<isize>(sh_str) {
                shift = sh;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            match roll_tensor(&tensor, shift, axis) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'roll\' requires a tensor or scalar value",
        )),
    }
}

fn roll_tensor(tensor: &Tensor, shift: isize, axis: usize) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r == 0 {
        return Ok(tensor.clone());
    }
    let dim_len = tensor.shape[axis];
    if dim_len == 0 {
        return Ok(tensor.clone());
    }

    let shift_norm = ((shift % dim_len as isize) + dim_len as isize) as usize % dim_len;
    if shift_norm == 0 {
        return Ok(tensor.clone());
    }

    let split_idx = dim_len - shift_norm;
    let part1 = tensor
        .slice_range(axis, split_idx, dim_len, 1)
        .map_err(|e| e.to_string())?;
    let part2 = tensor
        .slice_range(axis, 0, split_idx, 1)
        .map_err(|e| e.to_string())?;

    Tensor::concat(&[part1, part2], axis as isize).map_err(|e| e.to_string())
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
    core_types::shapecheck::axis_plan(result, rank, 1, Some(0), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_roll_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("shift"), RString::from("2")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.as_f32_slice().unwrap(), &[4.0, 5.0, 1.0, 2.0, 3.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_roll_accepts_a_scalar_value() {
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
