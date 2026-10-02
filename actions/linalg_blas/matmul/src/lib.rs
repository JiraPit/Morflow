mod interface {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../plugins/openblas/src/interface.rs"
    ));
}

use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload};

const OPERATION: u32 = interface::operation!(matmul);

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite | DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: core_types::ValueShape,
    args: A,
) -> core_types::ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::matmul(input, args)
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
        "matmul",
        get_input_type(),
        get_output_type(),
        None,
        Some(get_output_value_shape),
    )
}

/// Check this action's LP64 integer limits using shapes alone.
fn dimension_limits(input: &core_types::InputDescriptor) -> Result<(), core_types::RString> {
    fn check(value: &core_types::ValueShape) -> Result<(), core_types::RString> {
        if let Some(parts) = value.components() {
            for part in parts {
                check(part)?;
            }
        } else if let Some(shape) = value.shape() {
            if shape
                .dims()
                .iter()
                .rev()
                .take(2)
                .filter_map(|d| d.known())
                .any(|n| n > i32::MAX as usize)
            {
                return Err(
                    "matmul dimensions exceed the signed 32-bit LP64 OpenBLAS limit".into(),
                );
            }
        }
        Ok(())
    }
    check(&input.value)
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
    use core_types::Tensor;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute_with_context(
            "matmul",
            shapecheck,
            super::process,
            payload,
            Some(test_context()),
        )
    }
    use core_types::RVec;

    #[test]
    fn test_matmul_action() {
        // [2, 3] x [3, 2] -> [2, 2]
        let a = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_f32_shape(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0], vec![3, 2]).unwrap();

        let mut items = RVec::new();
        items.push(Payload::Tensor(a));
        items.push(Payload::Tensor(b));

        let res = process(Payload::Composite(items));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            // [1*7+2*9+3*2, 1*8+2*1+3*3] = [7+18+6, 8+2+9] = [31, 19]
            // [4*7+5*9+6*2, 4*8+5*1+6*3] = [28+45+12, 32+5+18] = [85, 55]
            assert_eq!(out.as_f32_slice().unwrap(), &[31.0, 19.0, 85.0, 55.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
