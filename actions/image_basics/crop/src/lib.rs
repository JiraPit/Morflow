use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, ImageLayout, Payload, Shape, ShapeResult};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg};
    let rank = input.rank();
    let result = contract::finish((|| {
        let (h, w, c, chw) = contract::image_dims(&input)?;
        let x = arg::<usize>(&args, &["x"], Some(0), Some(0))?.unwrap();
        let y = arg::<usize>(&args, &["y"], Some(1), Some(0))?.unwrap();
        let width = arg::<usize>(&args, &["width", "w"], Some(2), None)?;
        let height = arg::<usize>(&args, &["height", "h"], Some(3), None)?;
        let remaining_w = w.saturating_sub(x);
        let remaining_h = h.saturating_sub(y);
        let width = width.map_or(remaining_w, |n| {
            core_types::Dimension::Known(n).min(remaining_w)
        });
        let height = height.map_or(remaining_h, |n| {
            core_types::Dimension::Known(n).min(remaining_h)
        });
        if contract::image_layout_unknown(&input) {
            return Ok(Shape::unknown(input.rank()));
        }
        contract::image_shape(height, width, c, input.rank(), chw)
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
    let payload = match core_types::contract::image_input(payload, false) {
        Ok(payload) => payload,
        Err(error) => return Payload::Error(error),
    };
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));
    let mut crop_x = 0usize;
    let mut crop_y = 0usize;
    let mut crop_w: Option<usize> = None;
    let mut crop_h: Option<usize> = None;

    if let Some(args) = args_opt {
        if let Some(x_str) = args
            .get_named("x")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(x) = prepared.args.parse::<usize>(x_str) {
                crop_x = x;
            }
        }
        if let Some(y_str) = args
            .get_named("y")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(y) = prepared.args.parse::<usize>(y_str) {
                crop_y = y;
            }
        }
        if let Some(w_str) = args
            .get_named("width")
            .or_else(|| args.get_named("w"))
            .or_else(|| args.positional.get(2).map(|s| s.as_str()))
        {
            crop_w = prepared.args.parse::<usize>(w_str).ok();
        }
        if let Some(h_str) = args
            .get_named("height")
            .or_else(|| args.get_named("h"))
            .or_else(|| args.positional.get(3).map(|s| s.as_str()))
        {
            crop_h = prepared.args.parse::<usize>(h_str).ok();
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => {
            let shape = tensor.shape.as_slice();
            let layout = if shape.len() == 3 && shape[2] <= 4 {
                ImageLayout::Hwc
            } else if shape.len() == 3 && shape[0] <= 4 {
                ImageLayout::Chw
            } else {
                ImageLayout::Hwc
            };

            let (in_h, in_w) = match (shape.len(), layout) {
                (2, _) => (shape[0], shape[1]),
                (3, ImageLayout::Hwc) => (shape[0], shape[1]),
                (3, ImageLayout::Chw) => (shape[1], shape[2]),
                _ => return Payload::Tensor(tensor),
            };

            let x0 = crop_x.min(in_w);
            let y0 = crop_y.min(in_h);
            let w = crop_w.unwrap_or(in_w.saturating_sub(x0)).min(in_w - x0);
            let h = crop_h.unwrap_or(in_h.saturating_sub(y0)).min(in_h - y0);

            let (y_axis, x_axis) = match (shape.len(), layout) {
                (2, _) => (0, 1),
                (3, ImageLayout::Hwc) => (0, 1),
                (3, ImageLayout::Chw) => (1, 2),
                _ => (0, 1),
            };

            if let Ok(sliced_y) = tensor.slice_range(y_axis, y0, y0 + h, 1) {
                if let Ok(sliced_xy) = sliced_y.slice_range(x_axis, x0, x0 + w, 1) {
                    return Payload::Tensor(sliced_xy);
                }
            }

            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'crop\' requires Payload::Tensor",
        )),
    }
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
    fn test_crop_zero_copy() {
        let f32_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![4, 4]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("x"), RString::from("1")));
        named.push(Tuple2(RString::from("y"), RString::from("1")));
        named.push(Tuple2(RString::from("w"), RString::from("2")));
        named.push(Tuple2(RString::from("h"), RString::from("2")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[2, 2]);
            let contiguous = out_t.to_contiguous_bytes();
            let slice: &[f32] = unsafe {
                std::slice::from_raw_parts(contiguous.as_ptr() as *const f32, contiguous.len() / 4)
            };
            assert_eq!(slice, &[6.0, 7.0, 10.0, 11.0]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
