#![allow(clippy::too_many_arguments, clippy::manual_memcpy)]

use core_types::{
    ActionArgs, DataType, GetShapeFn, ImageLayout, Payload, Shape, Tensor, TensorDType,
};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> Shape {
    let dims_ = input.dims();
    let r = dims_.len();
    if r < 2 {
        return input;
    }
    let in_w = dims_[1];
    let in_h = dims_[0];
    let target_w: Option<usize> = args
        .get_named("width")
        .or_else(|| args.get_named("w"))
        .and_then(|s| s.parse::<usize>().ok());
    let target_h: Option<usize> = args
        .get_named("height")
        .or_else(|| args.get_named("h"))
        .and_then(|s| s.parse::<usize>().ok());
    let scale_x: Option<f32> = args
        .get_named("scale_x")
        .or_else(|| args.get_named("scale"))
        .and_then(|s| s.parse::<f32>().ok());
    let scale_y: Option<f32> = args
        .get_named("scale_y")
        .or_else(|| args.get_named("scale"))
        .and_then(|s| s.parse::<f32>().ok());
    let mut out_w = target_w.unwrap_or_else(|| {
        if in_w == 0 {
            return 0; // unknown dimension stays unknown
        }
        scale_x
            .map(|s| (in_w as f32 * s).round().max(1.0) as usize)
            .unwrap_or(in_w)
    });
    let mut out_h = target_h.unwrap_or_else(|| {
        if in_h == 0 {
            return 0; // unknown dimension stays unknown
        }
        scale_y
            .map(|s| (in_h as f32 * s).round().max(1.0) as usize)
            .unwrap_or(in_h)
    });
    if out_w == 0 && dims_[1] != 0 {
        out_w = 1;
    }
    if out_h == 0 && dims_[0] != 0 {
        out_h = 1;
    }
    let mut out = Vec::with_capacity(r);
    if r == 2 {
        out.push(out_h);
        out.push(out_w);
    } else {
        out.push(out_h);
        out.push(out_w);
        out.push(dims_[2]);
    }
    Shape::new(out)
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ResizeFilter {
    Bilinear,
    Nearest,
    Bicubic,
    Area,
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut target_w: Option<usize> = None;
    let mut target_h: Option<usize> = None;
    let mut scale_x: Option<f32> = None;
    let mut scale_y: Option<f32> = None;
    let mut filter = ResizeFilter::Bilinear;
    let mut keep_aspect_ratio = false;

    if let Some(args) = args_opt {
        if let Some(w_str) = args
            .get_named("width")
            .or_else(|| args.get_named("w"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            target_w = w_str.parse::<usize>().ok();
        }
        if let Some(h_str) = args
            .get_named("height")
            .or_else(|| args.get_named("h"))
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            target_h = h_str.parse::<usize>().ok();
        }
        if let Some(s_str) = args.get_named("scale") {
            if let Ok(s) = s_str.parse::<f32>() {
                scale_x = Some(s);
                scale_y = Some(s);
            }
        }
        if let Some(sx_str) = args.get_named("scale_x") {
            scale_x = sx_str.parse::<f32>().ok();
        }
        if let Some(sy_str) = args.get_named("scale_y") {
            scale_y = sy_str.parse::<f32>().ok();
        }
        if let Some(f_str) = args.get_named("filter") {
            filter = match f_str.to_lowercase().as_str() {
                "nearest" | "neighbor" => ResizeFilter::Nearest,
                "bicubic" | "cubic" => ResizeFilter::Bicubic,
                "area" | "box" => ResizeFilter::Area,
                _ => ResizeFilter::Bilinear,
            };
        }
        if let Some(ar_str) = args.get_named("keep_aspect_ratio") {
            keep_aspect_ratio = ar_str.to_lowercase() == "true" || ar_str == "1";
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => {
            let layout = if tensor.shape.len() == 3 && tensor.shape[2] <= 4 {
                ImageLayout::Hwc
            } else if tensor.shape.len() == 3 && tensor.shape[0] <= 4 {
                ImageLayout::Chw
            } else {
                ImageLayout::Hwc
            };
            let resized = resize_tensor(
                &tensor,
                layout,
                target_w,
                target_h,
                scale_x,
                scale_y,
                filter,
                keep_aspect_ratio,
            );
            Payload::Tensor(resized)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'resize\' requires Payload::Tensor",
        )),
    }
}

fn resize_tensor(
    tensor: &Tensor,
    layout: ImageLayout,
    target_w: Option<usize>,
    target_h: Option<usize>,
    scale_x: Option<f32>,
    scale_y: Option<f32>,
    filter: ResizeFilter,
    keep_aspect_ratio: bool,
) -> Tensor {
    let shape = tensor.shape.as_slice();
    let (in_h, in_w, channels) = match (shape.len(), layout) {
        (2, _) => (shape[0], shape[1], 1),
        (3, ImageLayout::Hwc) => (shape[0], shape[1], shape[2]),
        (3, ImageLayout::Chw) => (shape[1], shape[2], shape[0]),
        _ => return tensor.clone(),
    };

    if in_h == 0 || in_w == 0 {
        return tensor.clone();
    }

    let mut out_w = target_w.unwrap_or_else(|| {
        scale_x
            .map(|s| (in_w as f32 * s).round().max(1.0) as usize)
            .unwrap_or(in_w)
    });
    let mut out_h = target_h.unwrap_or_else(|| {
        scale_y
            .map(|s| (in_h as f32 * s).round().max(1.0) as usize)
            .unwrap_or(in_h)
    });

    if keep_aspect_ratio {
        let aspect = in_w as f32 / in_h as f32;
        if out_w as f32 / out_h as f32 > aspect {
            out_w = (out_h as f32 * aspect).round().max(1.0) as usize;
        } else {
            out_h = (out_w as f32 / aspect).round().max(1.0) as usize;
        }
    }

    if out_w == in_w && out_h == in_h {
        return tensor.clone();
    }

    match tensor.dtype {
        TensorDType::F32 => {
            let Some(src) = tensor.as_f32_slice() else {
                return tensor.clone();
            };
            let mut out = vec![0.0f32; out_h * out_w * channels];

            match layout {
                ImageLayout::Hwc => {
                    resize_hwc_f32(src, in_w, in_h, channels, &mut out, out_w, out_h, filter);
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![out_h, out_w]
                    } else {
                        vec![out_h, out_w, channels]
                    };
                    Tensor::from_f32_vec(out, out_shape).unwrap()
                }
                ImageLayout::Chw => {
                    resize_chw_f32(src, in_w, in_h, channels, &mut out, out_w, out_h, filter);
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![out_h, out_w]
                    } else {
                        vec![channels, out_h, out_w]
                    };
                    Tensor::from_f32_vec(out, out_shape).unwrap()
                }
            }
        }
        TensorDType::U8 => {
            let Some(bytes) = tensor.as_u8_slice() else {
                return tensor.clone();
            };
            let mut out = vec![0u8; out_h * out_w * channels];

            match layout {
                ImageLayout::Hwc => {
                    resize_hwc_u8(bytes, in_w, in_h, channels, &mut out, out_w, out_h, filter);
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![out_h, out_w]
                    } else {
                        vec![out_h, out_w, channels]
                    };
                    Tensor::from_rvec_u8(core_types::RVec::from(out), out_shape, TensorDType::U8)
                        .unwrap()
                }
                ImageLayout::Chw => {
                    resize_chw_u8(bytes, in_w, in_h, channels, &mut out, out_w, out_h, filter);
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![out_h, out_w]
                    } else {
                        vec![channels, out_h, out_w]
                    };
                    Tensor::from_rvec_u8(core_types::RVec::from(out), out_shape, TensorDType::U8)
                        .unwrap()
                }
            }
        }
        _ => tensor.clone(),
    }
}

fn resize_hwc_f32(
    src: &[f32],
    in_w: usize,
    in_h: usize,
    channels: usize,
    dst: &mut [f32],
    out_w: usize,
    out_h: usize,
    filter: ResizeFilter,
) {
    let scale_x = in_w as f32 / out_w as f32;
    let scale_y = in_h as f32 / out_h as f32;

    dst.par_chunks_exact_mut(out_w * channels)
        .enumerate()
        .for_each(|(out_y, row)| {
            for out_x in 0..out_w {
                let dst_idx = out_x * channels;
                match filter {
                    ResizeFilter::Nearest => {
                        let in_x = ((out_x as f32 * scale_x) as usize).min(in_w - 1);
                        let in_y = ((out_y as f32 * scale_y) as usize).min(in_h - 1);
                        let src_idx = (in_y * in_w + in_x) * channels;
                        for c in 0..channels {
                            row[dst_idx + c] = src[src_idx + c];
                        }
                    }
                    ResizeFilter::Bilinear => {
                        let center_x = (out_x as f32 + 0.5) * scale_x - 0.5;
                        let center_y = (out_y as f32 + 0.5) * scale_y - 0.5;

                        let x0 = (center_x.floor() as isize).clamp(0, in_w as isize - 1) as usize;
                        let x1 = (x0 + 1).min(in_w - 1);
                        let y0 = (center_y.floor() as isize).clamp(0, in_h as isize - 1) as usize;
                        let y1 = (y0 + 1).min(in_h - 1);

                        let dx = (center_x - x0 as f32).clamp(0.0, 1.0);
                        let dy = (center_y - y0 as f32).clamp(0.0, 1.0);

                        let w00 = (1.0 - dx) * (1.0 - dy);
                        let w10 = dx * (1.0 - dy);
                        let w01 = (1.0 - dx) * dy;
                        let w11 = dx * dy;

                        let idx00 = (y0 * in_w + x0) * channels;
                        let idx10 = (y0 * in_w + x1) * channels;
                        let idx01 = (y1 * in_w + x0) * channels;
                        let idx11 = (y1 * in_w + x1) * channels;

                        for c in 0..channels {
                            row[dst_idx + c] = w00 * src[idx00 + c]
                                + w10 * src[idx10 + c]
                                + w01 * src[idx01 + c]
                                + w11 * src[idx11 + c];
                        }
                    }
                    ResizeFilter::Bicubic => {
                        let center_x = (out_x as f32 + 0.5) * scale_x - 0.5;
                        let center_y = (out_y as f32 + 0.5) * scale_y - 0.5;

                        let x_int = center_x.floor() as isize;
                        let y_int = center_y.floor() as isize;
                        let dx = center_x - x_int as f32;
                        let dy = center_y - y_int as f32;

                        for c in 0..channels {
                            let mut val = 0.0f32;
                            for j in -1..=2 {
                                let py = (y_int + j).clamp(0, in_h as isize - 1) as usize;
                                let wy = cubic_weight(dy - j as f32);
                                for i in -1..=2 {
                                    let px = (x_int + i).clamp(0, in_w as isize - 1) as usize;
                                    let wx = cubic_weight(dx - i as f32);
                                    val += wx * wy * src[(py * in_w + px) * channels + c];
                                }
                            }
                            row[dst_idx + c] = val;
                        }
                    }
                    ResizeFilter::Area => {
                        let x_start = ((out_x as f32) * scale_x) as usize;
                        let x_end = (((out_x + 1) as f32 * scale_x).ceil() as usize).min(in_w);
                        let y_start = ((out_y as f32) * scale_y) as usize;
                        let y_end = (((out_y + 1) as f32 * scale_y).ceil() as usize).min(in_h);

                        let count = ((x_end - x_start) * (y_end - y_start)).max(1) as f32;
                        for c in 0..channels {
                            let mut sum = 0.0f32;
                            for sy in y_start..y_end {
                                for sx in x_start..x_end {
                                    sum += src[(sy * in_w + sx) * channels + c];
                                }
                            }
                            row[dst_idx + c] = sum / count;
                        }
                    }
                }
            }
        });
}

fn resize_chw_f32(
    src: &[f32],
    in_w: usize,
    in_h: usize,
    _channels: usize,
    dst: &mut [f32],
    out_w: usize,
    out_h: usize,
    filter: ResizeFilter,
) {
    let in_plane = in_h * in_w;
    let out_plane = out_h * out_w;
    let scale_x = in_w as f32 / out_w as f32;
    let scale_y = in_h as f32 / out_h as f32;

    dst.par_chunks_exact_mut(out_plane)
        .enumerate()
        .for_each(|(c, plane)| {
            let src_plane = &src[c * in_plane..(c + 1) * in_plane];
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let dst_idx = out_y * out_w + out_x;
                    match filter {
                        ResizeFilter::Nearest => {
                            let in_x = ((out_x as f32 * scale_x) as usize).min(in_w - 1);
                            let in_y = ((out_y as f32 * scale_y) as usize).min(in_h - 1);
                            plane[dst_idx] = src_plane[in_y * in_w + in_x];
                        }
                        ResizeFilter::Bilinear => {
                            let center_x = (out_x as f32 + 0.5) * scale_x - 0.5;
                            let center_y = (out_y as f32 + 0.5) * scale_y - 0.5;

                            let x0 =
                                (center_x.floor() as isize).clamp(0, in_w as isize - 1) as usize;
                            let x1 = (x0 + 1).min(in_w - 1);
                            let y0 =
                                (center_y.floor() as isize).clamp(0, in_h as isize - 1) as usize;
                            let y1 = (y0 + 1).min(in_h - 1);

                            let dx = (center_x - x0 as f32).clamp(0.0, 1.0);
                            let dy = (center_y - y0 as f32).clamp(0.0, 1.0);

                            plane[dst_idx] = (1.0 - dx) * (1.0 - dy) * src_plane[y0 * in_w + x0]
                                + dx * (1.0 - dy) * src_plane[y0 * in_w + x1]
                                + (1.0 - dx) * dy * src_plane[y1 * in_w + x0]
                                + dx * dy * src_plane[y1 * in_w + x1];
                        }
                        _ => {
                            let in_x = ((out_x as f32 * scale_x) as usize).min(in_w - 1);
                            let in_y = ((out_y as f32 * scale_y) as usize).min(in_h - 1);
                            plane[dst_idx] = src_plane[in_y * in_w + in_x];
                        }
                    }
                }
            }
        });
}

fn resize_hwc_u8(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    channels: usize,
    dst: &mut [u8],
    out_w: usize,
    out_h: usize,
    filter: ResizeFilter,
) {
    let scale_x = in_w as f32 / out_w as f32;
    let scale_y = in_h as f32 / out_h as f32;

    dst.par_chunks_exact_mut(out_w * channels)
        .enumerate()
        .for_each(|(out_y, row)| {
            for out_x in 0..out_w {
                let dst_idx = out_x * channels;
                match filter {
                    ResizeFilter::Nearest => {
                        let in_x = ((out_x as f32 * scale_x) as usize).min(in_w - 1);
                        let in_y = ((out_y as f32 * scale_y) as usize).min(in_h - 1);
                        let src_idx = (in_y * in_w + in_x) * channels;
                        for c in 0..channels {
                            row[dst_idx + c] = src[src_idx + c];
                        }
                    }
                    _ => {
                        let center_x = (out_x as f32 + 0.5) * scale_x - 0.5;
                        let center_y = (out_y as f32 + 0.5) * scale_y - 0.5;

                        let x0 = (center_x.floor() as isize).clamp(0, in_w as isize - 1) as usize;
                        let x1 = (x0 + 1).min(in_w - 1);
                        let y0 = (center_y.floor() as isize).clamp(0, in_h as isize - 1) as usize;
                        let y1 = (y0 + 1).min(in_h - 1);

                        let dx = (center_x - x0 as f32).clamp(0.0, 1.0);
                        let dy = (center_y - y0 as f32).clamp(0.0, 1.0);

                        let w00 = (1.0 - dx) * (1.0 - dy);
                        let w10 = dx * (1.0 - dy);
                        let w01 = (1.0 - dx) * dy;
                        let w11 = dx * dy;

                        let idx00 = (y0 * in_w + x0) * channels;
                        let idx10 = (y0 * in_w + x1) * channels;
                        let idx01 = (y1 * in_w + x0) * channels;
                        let idx11 = (y1 * in_w + x1) * channels;

                        for c in 0..channels {
                            let val = w00 * src[idx00 + c] as f32
                                + w10 * src[idx10 + c] as f32
                                + w01 * src[idx01 + c] as f32
                                + w11 * src[idx11 + c] as f32;
                            row[dst_idx + c] = val.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
            }
        });
}

fn resize_chw_u8(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    _channels: usize,
    dst: &mut [u8],
    out_w: usize,
    out_h: usize,
    filter: ResizeFilter,
) {
    let in_plane = in_h * in_w;
    let out_plane = out_h * out_w;
    let scale_x = in_w as f32 / out_w as f32;
    let scale_y = in_h as f32 / out_h as f32;

    dst.par_chunks_exact_mut(out_plane)
        .enumerate()
        .for_each(|(c, plane)| {
            let src_plane = &src[c * in_plane..(c + 1) * in_plane];
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let dst_idx = out_y * out_w + out_x;
                    match filter {
                        ResizeFilter::Nearest => {
                            let in_x = ((out_x as f32 * scale_x) as usize).min(in_w - 1);
                            let in_y = ((out_y as f32 * scale_y) as usize).min(in_h - 1);
                            plane[dst_idx] = src_plane[in_y * in_w + in_x];
                        }
                        _ => {
                            let center_x = (out_x as f32 + 0.5) * scale_x - 0.5;
                            let center_y = (out_y as f32 + 0.5) * scale_y - 0.5;

                            let x0 =
                                (center_x.floor() as isize).clamp(0, in_w as isize - 1) as usize;
                            let x1 = (x0 + 1).min(in_w - 1);
                            let y0 =
                                (center_y.floor() as isize).clamp(0, in_h as isize - 1) as usize;
                            let y1 = (y0 + 1).min(in_h - 1);

                            let dx = (center_x - x0 as f32).clamp(0.0, 1.0);
                            let dy = (center_y - y0 as f32).clamp(0.0, 1.0);

                            let val = (1.0 - dx) * (1.0 - dy) * src_plane[y0 * in_w + x0] as f32
                                + dx * (1.0 - dy) * src_plane[y0 * in_w + x1] as f32
                                + (1.0 - dx) * dy * src_plane[y1 * in_w + x0] as f32
                                + dx * dy * src_plane[y1 * in_w + x1] as f32;
                            plane[dst_idx] = val.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
            }
        });
}

#[inline]
fn cubic_weight(x: f32) -> f32 {
    let a = -0.5f32;
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        (a + 2.0) * abs_x * abs_x * abs_x - (a + 3.0) * abs_x * abs_x + 1.0
    } else if abs_x < 2.0 {
        a * abs_x * abs_x * abs_x - 5.0 * a * abs_x * abs_x + 8.0 * a * abs_x - 4.0 * a
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_resize_upscale_bilinear() {
        let f32_data = vec![0.0f32, 1.0, 1.0, 0.0];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![2, 2, 1]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("width"), RString::from("4")));
        named.push(Tuple2(RString::from("height"), RString::from("4")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[4, 4, 1]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
