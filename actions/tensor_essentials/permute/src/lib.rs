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
    use core_types::contract::{self, Error};
    let rank = input.rank();
    let result = contract::finish((|| {
        let text = contract::value(&args, &["dims"], Some(0))?
            .ok_or(Error::from("permute requires 'dims'"))?;
        let dims = args.usize_list(text).map_err(Error::from)?;
        if dims.len() != input.rank() {
            return Err("Permutation length must match input rank".into());
        }
        let mut seen = vec![false; input.rank()];
        let mut out = Vec::new();
        for d in dims {
            if d >= input.rank() || seen[d] {
                return Err("Permutation must contain each input axis exactly once".into());
            }
            seen[d] = true;
            out.push(input.dims()[d]);
        }
        contract::shape(out)
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

    let mut dims_str = None;
    if let Some(args) = &args_opt {
        dims_str = args
            .get_named("dims")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    let Some(d_str) = dims_str else {
        return Payload::Error("permute action requires 'dims' argument".into());
    };

    let dims = match prepared.args.usize_list(d_str) {
        Ok(d) => d,
        Err(e) => return Payload::Error(e),
    };

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match tensor.permute(&dims) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e),
        },
        _ => Payload::Error(core_types::RString::from(
            "Action \'permute\' requires a tensor or scalar value",
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
    fn test_permute_action() {
        let tensor =
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
                .unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dims"), RString::from("[2, 0, 1]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2, 2]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_permute_accepts_a_scalar_value() {
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dims"), RString::from("[]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::scalar_f32(2.0)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };
        let res = process(payload);
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
