mod interface {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../plugins/openblas/src/interface.rs"
    ));
}

use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};

const OPERATION: u32 = interface::operation!(cholesky);
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if input.rank() != 2 {
            return Err("cholesky requires a rank-2 matrix".into());
        }
        let rank = input.rank();
        let (h, w) = (input.dims()[rank - 2], input.dims()[rank - 1]);
        if !h.is_unknown() && !w.is_unknown() && h != w {
            return Err("Matrix must be square".into());
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
    call_plugin(OPERATION, payload, prepared)
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    if let Err(reason) = dimension_limits(&input, "cholesky") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "cholesky",
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

/// Validate the backend's integer ABI from shapes alone, without loading it.
fn dimension_limits(
    input: &core_types::InputDescriptor,
    action: &str,
) -> Result<(), core_types::RString> {
    fn check(value: &core_types::ValueShape, action: &str) -> Result<(), core_types::RString> {
        if let Some(parts) = value.components() {
            for part in parts {
                check(part, action)?;
            }
        } else if let Some(shape) = value.shape() {
            let too_large = if matches!(action, "dot" | "outer") {
                shape
                    .known_dims()
                    .and_then(|dims| dims.iter().try_fold(1usize, |n, d| n.checked_mul(*d)))
                    .is_some_and(|n| n > i32::MAX as usize)
            } else {
                shape
                    .dims()
                    .iter()
                    .rev()
                    .take(2)
                    .filter_map(|d| d.known())
                    .any(|n| n > i32::MAX as usize)
            };
            if too_large {
                return Err(format!(
                    "{action} dimensions exceed the signed 32-bit LP64 OpenBLAS limit"
                )
                .into());
            }
        }
        Ok(())
    }
    check(&input.value, action)
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
            "cholesky",
            shapecheck,
            super::process,
            payload,
            Some(test_context()),
        )
    }
    use core_types::Tensor;

    #[test]
    fn test_cholesky_action() {
        // A = [4, 12, -16; 12, 37, -43; -16, -43, 98]
        // L = [2, 0, 0; 6, 1, 0; -8, 5, 3]
        let a = Tensor::from_f32_shape(
            &[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
            vec![3, 3],
        )
        .unwrap();

        let res = process(Payload::Tensor(a));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[3, 3]);
            let slice = out.as_f32_slice().unwrap();
            assert_eq!(slice, &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0,]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
