mod interface {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../plugins/openblas/src/interface.rs"
    ));
}

use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};

const OPERATION: u32 = interface::operation!(inv);

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
        if input.rank() < 2 {
            return Err("inv requires rank at least 2".into());
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
    if let Err(reason) = dimension_limits(&input) {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "inv",
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    )
}

/// Check this action's LP64 integer limits using shapes alone.
fn dimension_limits(input: &core_types::InputDescriptor) -> Result<(), core_types::RString> {
    if let Some(shape) = input.value.shape() {
        if shape
            .dims()
            .iter()
            .rev()
            .take(2)
            .filter_map(|d| d.known())
            .any(|n| n > i32::MAX as usize)
        {
            return Err("inv dimensions exceed the signed 32-bit LP64 OpenBLAS limit".into());
        }
    }
    Ok(())
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
            "inv",
            shapecheck,
            super::process,
            payload,
            Some(test_context()),
        )
    }
    use core_types::Tensor;

    #[test]
    fn test_inv_action() {
        // [4, 7; 2, 6] -> det = 24 - 14 = 10 -> inv = [0.6, -0.7; -0.2, 0.4]
        let mat = Tensor::from_f32_shape(&[4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            assert!((slice[0] - 0.6).abs() < 1e-4);
            assert!((slice[1] - (-0.7)).abs() < 1e-4);
            assert!((slice[2] - (-0.2)).abs() < 1e-4);
            assert!((slice[3] - 0.4).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
