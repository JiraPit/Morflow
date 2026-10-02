use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};
fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if !matches!(input.rank(), 2 | 3) {
            return Err("Image operations require rank 2 or 3".into());
        }
        contract::image_nonempty(&input)?;
        Ok(input)
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}
#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}
#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    morflow_opencv::analyze(input, args, "sharpen", get_output_shape)
}
#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    morflow_opencv::process("sharpen", payload, prepared)
}

#[no_mangle]
pub extern "C" fn get_required_plugins() -> core_types::RVec<core_types::plugins::PluginRequirement>
{
    morflow_opencv::required_plugins()
}
