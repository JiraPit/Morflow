use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};
fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg};
    let rank = input.rank();
    let result = contract::finish((|| {
        let (h, w, c, chw) = contract::image_dims(&input)?;
        let angle = arg::<f32>(&args, &["angle", "angle_deg"], Some(0), Some(90.0))?.unwrap();
        if !angle.is_finite() {
            return Err("Rotation angle must be finite".into());
        }
        let expand = contract::value(&args, &["expand", "expand_canvas"], None)?
            .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
            .unwrap_or(true);
        let norm = ((angle % 360.0) + 360.0) % 360.0;
        let (oh, ow) = if (norm - 90.0).abs() < 1e-3 || (norm - 270.0).abs() < 1e-3 {
            (w, h)
        } else if norm.abs() < 1e-3 || (norm - 180.0).abs() < 1e-3 || !expand {
            (h, w)
        } else if h.is_unknown() || w.is_unknown() {
            (
                core_types::Dimension::Unknown,
                core_types::Dimension::Unknown,
            )
        } else {
            let (h, w) = (h.known().unwrap(), w.known().unwrap());
            let rad = norm.to_radians();
            let sin = rad.sin().abs();
            let cos = rad.cos().abs();
            let ow = (w as f32 * cos + h as f32 * sin).round().max(1.0);
            let oh = (w as f32 * sin + h as f32 * cos).round().max(1.0);
            if !ow.is_finite()
                || !oh.is_finite()
                || ow >= usize::MAX as f32
                || oh >= usize::MAX as f32
            {
                return Err("Rotated dimensions overflow".into());
            }
            ((oh as usize).into(), (ow as usize).into())
        };
        contract::image_shape(oh, ow, c, input.rank(), chw)
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
    morflow_opencv::analyze(input, args, "rotate", get_output_shape)
}
#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    morflow_opencv::process("rotate", payload, prepared)
}

#[no_mangle]
pub extern "C" fn get_required_plugins() -> core_types::RVec<core_types::plugins::PluginRequirement>
{
    morflow_opencv::required_plugins()
}
