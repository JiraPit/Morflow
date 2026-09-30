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
    use core_types::contract::{self, Error};
    contract::finish((|| {
        let text = contract::value(&args, &["dims"], Some(0))?
            .ok_or(Error::from("permute requires 'dims'"))?;
        let dims = parse_dims_str(text).map_err(Error::from)?;
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
    })())
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
}

fn parse_dims_str(s: &str) -> Result<Vec<usize>, String> {
    let clean = s
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_start_matches('(')
        .trim_end_matches(')');
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    clean
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| format!("Invalid dimension '{}': {}", p, e))
        })
        .collect()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut dims_str = None;
    if let Some(args) = &args_opt {
        dims_str = args
            .get_named("dims")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    let Some(d_str) = dims_str else {
        return Payload::Error("permute action requires 'dims' argument".into());
    };

    let dims = match parse_dims_str(d_str) {
        Ok(d) => d,
        Err(e) => return Payload::Error(e.into()),
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

#[cfg(test)]
mod tests {
    use super::*;
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
