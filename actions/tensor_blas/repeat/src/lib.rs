mod interface {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../plugins/openblas/src/interface.rs"
    ));
}

use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};

const OPERATION: u32 = interface::operation!(repeat);
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
    contract::finish((|| {
        let text = contract::value(&args, &["repeats"], Some(0))?
            .ok_or(Error::from("repeat requires 'repeats'"))?;
        let repeats = args.usize_list(text).map_err(Error::from)?;
        let rank = input.rank().max(repeats.len());
        let mut out = vec![core_types::Dimension::Known(1); rank - input.rank()];
        out.extend_from_slice(input.dims());
        let offset = rank - repeats.len();
        for (i, repeat) in repeats.iter().enumerate() {
            out[offset + i] = out[offset + i]
                .checked_mul((*repeat).max(1))
                .ok_or(Error::from("Repeated dimension overflows"))?;
        }
        contract::shape(out)
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    call_plugin(OPERATION, payload, prepared)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    core_types::shapecheck::analyze(
        input,
        args,
        "repeat",
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

fn call_plugin(operation: u32, payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let Some(runtime) = prepared.runtime.as_ref().into_option() else {
        return Payload::Error("OpenBLAS action requires an engine-provided plugin context".into());
    };
    let address = match runtime.symbol(interface::PLUGIN_NAME, interface::PROCESS_SYMBOL) {
        Ok(address) => address,
        Err(error) => return Payload::Error(error),
    };
    // The openblas plugin defines this exact ABI. All values use the shared SDK.
    let process = unsafe { std::mem::transmute::<usize, interface::NativeProcess>(address) };
    process(operation, payload, prepared)
}

#[no_mangle]
pub extern "C" fn get_required_plugins() -> core_types::RVec<core_types::plugins::PluginRequirement>
{
    vec![core_types::plugins::PluginRequirement {
        name: interface::PLUGIN_NAME.into(),
        version: interface::PLUGIN_VERSION.into(),
    }]
    .into()
}

#[cfg(test)]
fn test_context() -> core_types::plugins::RuntimeContext {
    extern "C" fn release(_: usize) {}
    extern "C" fn resolve(
        _: usize,
        plugin: core_types::RString,
        symbol: core_types::RString,
    ) -> core_types::abi_stable::std_types::RResult<usize, core_types::RString> {
        if plugin == interface::PLUGIN_NAME && symbol == interface::PROCESS_SYMBOL {
            core_types::abi_stable::std_types::RResult::ROk(
                openblas_plugin::morflow_openblas_process as *const () as usize,
            )
        } else {
            core_types::abi_stable::std_types::RResult::RErr("Unexpected plugin symbol".into())
        }
    }
    // Test-only context calls the real plugin export with the same native interface.
    unsafe { core_types::plugins::RuntimeContext::new(0, release, resolve) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute_with_context(
            "repeat",
            shapecheck,
            super::process,
            payload,
            Some(test_context()),
        )
    }
    use core_types::{ActionArgs, RBox, RString, Tensor, Tuple2};

    #[test]
    fn test_repeat_action() {
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0], vec![1, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("repeats"), RString::from("[2, 1]")));
        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let res = process(payload);
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
    #[test]
    fn test_repeat_accepts_a_scalar_value() {
        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("repeats"), RString::from("[]")));
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
