use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult};
fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg};
    let rank = input.rank();
    let result = contract::finish((|| {
        let (h, w, c, chw) = contract::image_dims(&input)?;
        let width = arg::<usize>(&args, &["width", "w"], Some(0), None)?;
        let height = arg::<usize>(&args, &["height", "h"], Some(1), None)?;
        let sx = arg::<f32>(&args, &["scale_x", "scale"], None, None)?;
        let sy = arg::<f32>(&args, &["scale_y", "scale"], None, None)?;
        if sx.into_iter().chain(sy).any(|s| !s.is_finite() || s <= 0.0) {
            return Err("Resize scales must be positive and finite".into());
        }
        let aspect = contract::value(&args, &["keep_aspect_ratio"], None)?
            .is_some_and(|s| s.eq_ignore_ascii_case("true") || s == "1");
        let scale = |dim: core_types::Dimension,
                     factor: Option<f32>|
         -> core_types::contract::Result<core_types::Dimension> {
            let (Some(n), Some(s)) = (dim.known(), factor) else {
                return Ok(dim);
            };
            let n = (n as f32 * s).round().max(1.0);
            if !n.is_finite() || n >= usize::MAX as f32 {
                return Err("Resized dimension overflows".into());
            }
            Ok(core_types::Dimension::Known(n as usize))
        };
        let mut ow = match width {
            Some(n) => n.into(),
            None => scale(w, sx)?,
        };
        let mut oh = match height {
            Some(n) => n.into(),
            None => scale(h, sy)?,
        };
        if ow == 0 || oh == 0 {
            return Err("Resize dimensions must be positive".into());
        }
        if aspect {
            if let (Some(w), Some(h), Some(out_w), Some(out_h)) =
                (w.known(), h.known(), ow.known(), oh.known())
            {
                let ratio = w as f32 / h as f32;
                if out_w as f32 / out_h as f32 > ratio {
                    ow = core_types::Dimension::Known(
                        (out_h as f32 * ratio).round().max(1.0) as usize
                    );
                } else {
                    oh = core_types::Dimension::Known(
                        (out_w as f32 / ratio).round().max(1.0) as usize
                    );
                }
            } else {
                ow = core_types::Dimension::Unknown;
                oh = core_types::Dimension::Unknown;
            }
        }
        if contract::image_layout_unknown(&input) {
            return Ok(Shape::unknown(input.rank()));
        }
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
    morflow_opencv::analyze(input, args, "resize", get_output_shape)
}
#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    morflow_opencv::process("resize", payload, prepared)
}

#[no_mangle]
pub extern "C" fn get_required_plugins() -> core_types::RVec<core_types::plugins::PluginRequirement>
{
    morflow_opencv::required_plugins()
}
