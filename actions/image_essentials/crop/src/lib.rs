use core_types::{
    ActionArgs, DataType, GetShapeResultFn, ImageLayout, Payload, Shape, ShapeResult,
};
#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn shape_impl(input: Shape, args: ActionArgs) -> Shape {
    let dims_ = input.dims();
    let r = dims_.len();
    if r < 2 {
        return input;
    }
    let (in_h, in_w) = (dims_[0], dims_[1]);
    let mut crop_x = 0usize;
    let mut crop_y = 0usize;
    let mut crop_w: Option<usize> = None;
    let mut crop_h: Option<usize> = None;
    if let Some(x_str) = args
        .get_named("x")
        .or_else(|| args.positional.first().map(|s| s.as_str()))
    {
        if let Ok(v) = x_str.parse::<usize>() {
            crop_x = v;
        }
    }
    if let Some(y_str) = args
        .get_named("y")
        .or_else(|| args.positional.get(1).map(|s| s.as_str()))
    {
        if let Ok(v) = y_str.parse::<usize>() {
            crop_y = v;
        }
    }
    if let Some(w_str) = args
        .get_named("width")
        .or_else(|| args.get_named("w"))
        .or_else(|| args.positional.get(2).map(|s| s.as_str()))
    {
        crop_w = w_str.parse::<usize>().ok();
    }
    if let Some(h_str) = args
        .get_named("height")
        .or_else(|| args.get_named("h"))
        .or_else(|| args.positional.get(3).map(|s| s.as_str()))
    {
        crop_h = h_str.parse::<usize>().ok();
    }
    let x0 = crop_x.min(in_w);
    let y0 = crop_y.min(in_h);
    let w = crop_w.unwrap_or(in_w.saturating_sub(x0)).min(in_w - x0);
    let h = crop_h.unwrap_or(in_h.saturating_sub(y0)).min(in_h - y0);
    let mut out = Vec::with_capacity(r);
    if r == 3 && in_w > 4 && dims_[0] <= 4 {
        // CHW layout: [C, H, W]
        out.push(dims_[0]);
        out.push(h);
        out.push(w);
    } else {
        out.push(h);
        out.push(w);
        if r == 3 {
            out.push(dims_[2]);
        }
    }
    Shape::new(out)
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeResultFn = get_output_shape_result;

#[no_mangle]
pub extern "C" fn get_output_shape_result(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut crop_x = 0usize;
    let mut crop_y = 0usize;
    let mut crop_w: Option<usize> = None;
    let mut crop_h: Option<usize> = None;

    if let Some(args) = args_opt {
        if let Some(x_str) = args
            .get_named("x")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(x) = x_str.parse::<usize>() {
                crop_x = x;
            }
        }
        if let Some(y_str) = args
            .get_named("y")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(y) = y_str.parse::<usize>() {
                crop_y = y;
            }
        }
        if let Some(w_str) = args
            .get_named("width")
            .or_else(|| args.get_named("w"))
            .or_else(|| args.positional.get(2).map(|s| s.as_str()))
        {
            crop_w = w_str.parse::<usize>().ok();
        }
        if let Some(h_str) = args
            .get_named("height")
            .or_else(|| args.get_named("h"))
            .or_else(|| args.positional.get(3).map(|s| s.as_str()))
        {
            crop_h = h_str.parse::<usize>().ok();
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

#[cfg(test)]
mod tests {
    use super::*;
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
