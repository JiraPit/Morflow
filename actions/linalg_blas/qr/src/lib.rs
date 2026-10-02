mod interface {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../plugins/openblas/src/interface.rs"
    ));
}

use core_types::shapecheck::PreparedArgs;
use core_types::{ActionArgs, Shape, ShapeResult};

const OPERATION: u32 = interface::operation!(qr);
use core_types::{DataType, Payload, RVec};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Composite
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, _args: A) -> ShapeResult {
    let _args = _args.into();
    if input.rank() != 2 {
        return ShapeResult::Invalid("qr received an unsupported input rank".into());
    }
    if input.dims()[1].checked_mul(input.dims()[1]).is_none() {
        return ShapeResult::Invalid("QR output matrix element count overflows".into());
    }
    ShapeResult::Unknown
}

pub extern "C" fn get_output_components(
    input: Shape,
    args: ActionArgs,
) -> RVec<core_types::OutputComponent> {
    if matches!(
        get_output_shape(input.clone(), args),
        ShapeResult::Invalid(_)
    ) {
        return RVec::new();
    }
    let n = input.dims()[1];
    vec![
        core_types::OutputComponent {
            kind: DataType::Tensor,
            shape: ShapeResult::Ok(input),
        },
        core_types::OutputComponent {
            kind: DataType::Tensor,
            shape: ShapeResult::Ok(Shape::new([n, n])),
        },
    ]
    .into()
}

pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: core_types::ValueShape,
    args: A,
) -> core_types::ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::qr(input, args)
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
    if let Err(reason) = dimension_limits(&input, "qr") {
        return core_types::ShapeCheckResult::Invalid { reason };
    }
    core_types::shapecheck::analyze(
        input,
        args,
        "qr",
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        Some(get_output_value_shape),
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
            "qr",
            shapecheck,
            super::process,
            payload,
            Some(test_context()),
        )
    }
    use core_types::Tensor;

    #[test]
    fn test_qr_action() {
        let mat = Tensor::from_f32_shape(
            &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            vec![3, 3],
        )
        .unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Composite(items) = res {
            assert_eq!(items.len(), 2);
            if let (Payload::Tensor(q), Payload::Tensor(r)) = (&items[0], &items[1]) {
                assert_eq!(q.shape.as_slice(), &[3, 3]);
                assert_eq!(r.shape.as_slice(), &[3, 3]);
            } else {
                panic!("Expected Tensor Q and R");
            }
        } else {
            panic!("Expected Composite output");
        }
    }
}
