use core_types::{ColorSpace, DataType, ImageLayout, Payload, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TargetDType {
    F32,
    U8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TargetLayout {
    Hwc,
    Chw,
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut target_color: Option<ColorSpace> = None;
    let mut target_dtype = TargetDType::F32;
    let mut target_layout = TargetLayout::Hwc;
    let mut normalize: Option<bool> = None;

    if let Some(args) = payload.args() {
        if let Some(c) = args
            .get_named("color")
            .or_else(|| args.get_named("color_space"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            target_color = match c.to_lowercase().as_str() {
                "rgb" => Some(ColorSpace::Rgb),
                "rgba" => Some(ColorSpace::Rgba),
                "bgr" => Some(ColorSpace::Bgr),
                "bgra" => Some(ColorSpace::Bgra),
                "grayscale" | "gray" | "grey" => Some(ColorSpace::Grayscale),
                _ => None,
            };
        }
        if let Some(dt) = args.get_named("dtype") {
            match dt.to_lowercase().as_str() {
                "f32" | "float" | "float32" => target_dtype = TargetDType::F32,
                "u8" | "uint8" | "byte" | "bytes" => target_dtype = TargetDType::U8,
                _ => {}
            }
        }
        if let Some(lay) = args.get_named("layout") {
            match lay.to_lowercase().as_str() {
                "chw" | "planar" => target_layout = TargetLayout::Chw,
                "hwc" | "interleaved" => target_layout = TargetLayout::Hwc,
                _ => {}
            }
        }
        if let Some(norm_str) = args.get_named("normalize") {
            normalize = match norm_str.to_lowercase().as_str() {
                "true" | "1" | "yes" => Some(true),
                "false" | "0" | "no" => Some(false),
                _ => None,
            };
        }
    }

    match payload.unwrap_payload() {
        Payload::Image(img) => {
            let tensor = standardize_image_to_tensor(
                img,
                target_color,
                target_dtype,
                target_layout,
                normalize,
            );
            Payload::Tensor(tensor)
        }
        Payload::Tensor(tensor) => {
            let tensor = standardize_raw_tensor(
                tensor,
                target_color,
                target_dtype,
                target_layout,
                normalize,
            );
            Payload::Tensor(tensor)
        }
        Payload::Audio(audio) => Payload::Tensor(audio.tensor.clone()),
        other => other.clone(),
    }
}

fn standardize_image_to_tensor(
    img: &core_types::Image,
    target_color: Option<ColorSpace>,
    target_dtype: TargetDType,
    target_layout: TargetLayout,
    normalize: Option<bool>,
) -> Tensor {
    let width = img.width();
    let height = img.height();
    let src_color = img.color_space;
    let dst_color = target_color.unwrap_or(src_color);
    let should_norm = normalize.unwrap_or(true);

    // O(1) Zero-Copy Fast-Path: if already matching requested format, return tensor directly
    let cur_target_layout = if img.layout == ImageLayout::Hwc {
        TargetLayout::Hwc
    } else {
        TargetLayout::Chw
    };
    let cur_target_dtype = if img.dtype() == TensorDType::F32 {
        TargetDType::F32
    } else {
        TargetDType::U8
    };
    if src_color == dst_color
        && cur_target_layout == target_layout
        && cur_target_dtype == target_dtype
    {
        if (target_dtype == TargetDType::F32 && img.dtype() == TensorDType::F32)
            || (target_dtype == TargetDType::U8 && img.dtype() == TensorDType::U8 && !should_norm)
        {
            return img.tensor.clone();
        }
    }

    // Convert pixel data to F32 intermediate in HWC
    let hwc_f32: Vec<f32> = match img.dtype() {
        TensorDType::U8 => {
            let bytes = img.to_contiguous_bytes();
            let src_channels = src_color.channels();
            let dst_channels = dst_color.channels();
            let mut out = vec![0.0f32; height * width * dst_channels];

            let scale = if should_norm { 1.0 / 255.0 } else { 1.0 };

            out.par_chunks_exact_mut(width * dst_channels)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..width {
                        let (r, g, b, a) = match img.layout {
                            ImageLayout::Hwc => {
                                let idx = (y * width + x) * src_channels;
                                extract_rgba_u8(&bytes[idx..idx + src_channels], src_color)
                            }
                            ImageLayout::Chw => {
                                let plane_size = height * width;
                                let pixel_idx = y * width + x;
                                extract_rgba_u8_chw(&bytes, pixel_idx, plane_size, src_color)
                            }
                        };

                        let dst_idx = x * dst_channels;
                        write_rgba_f32(
                            &mut row[dst_idx..dst_idx + dst_channels],
                            dst_color,
                            r * scale,
                            g * scale,
                            b * scale,
                            a * scale,
                        );
                    }
                });
            out
        }
        TensorDType::F32 => {
            let bytes = img.to_contiguous_bytes();
            let src_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            let src_channels = src_color.channels();
            let dst_channels = dst_color.channels();
            let mut out = vec![0.0f32; height * width * dst_channels];

            out.par_chunks_exact_mut(width * dst_channels)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..width {
                        let (r, g, b, a) = match img.layout {
                            ImageLayout::Hwc => {
                                let idx = (y * width + x) * src_channels;
                                extract_rgba_f32(&src_f32[idx..idx + src_channels], src_color)
                            }
                            ImageLayout::Chw => {
                                let plane_size = height * width;
                                let pixel_idx = y * width + x;
                                extract_rgba_f32_chw(src_f32, pixel_idx, plane_size, src_color)
                            }
                        };

                        let dst_idx = x * dst_channels;
                        write_rgba_f32(
                            &mut row[dst_idx..dst_idx + dst_channels],
                            dst_color,
                            r,
                            g,
                            b,
                            a,
                        );
                    }
                });
            out
        }
        _ => return img.tensor.clone(),
    };

    let dst_channels = dst_color.channels();

    // Now convert to target dtype and layout
    match (target_dtype, target_layout) {
        (TargetDType::F32, TargetLayout::Hwc) => {
            let shape = if dst_channels == 1 {
                vec![height, width]
            } else {
                vec![height, width, dst_channels]
            };
            Tensor::from_f32_shape(&hwc_f32, shape).unwrap()
        }
        (TargetDType::F32, TargetLayout::Chw) => {
            let shape = if dst_channels == 1 {
                vec![height, width]
            } else {
                vec![dst_channels, height, width]
            };
            if dst_channels == 1 {
                Tensor::from_f32_shape(&hwc_f32, shape).unwrap()
            } else {
                let mut chw = vec![0.0f32; dst_channels * height * width];
                let plane_size = height * width;
                chw.par_chunks_exact_mut(plane_size)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for y in 0..height {
                            for x in 0..width {
                                plane[y * width + x] = hwc_f32[(y * width + x) * dst_channels + c];
                            }
                        }
                    });
                Tensor::from_f32_shape(&chw, shape).unwrap()
            }
        }
        (TargetDType::U8, TargetLayout::Hwc) => {
            let shape = if dst_channels == 1 {
                vec![height, width]
            } else {
                vec![height, width, dst_channels]
            };
            let mut u8_data = vec![0u8; height * width * dst_channels];
            let multiplier = if should_norm { 255.0 } else { 1.0 };
            u8_data
                .par_iter_mut()
                .zip(hwc_f32.par_iter())
                .for_each(|(dst, &src)| {
                    *dst = (src * multiplier).clamp(0.0, 255.0).round() as u8;
                });
            Tensor::from_rvec_u8(core_types::RVec::from(u8_data), shape, TensorDType::U8).unwrap()
        }
        (TargetDType::U8, TargetLayout::Chw) => {
            let shape = if dst_channels == 1 {
                vec![height, width]
            } else {
                vec![dst_channels, height, width]
            };
            let multiplier = if should_norm { 255.0 } else { 1.0 };
            if dst_channels == 1 {
                let mut u8_data = vec![0u8; height * width];
                u8_data
                    .par_iter_mut()
                    .zip(hwc_f32.par_iter())
                    .for_each(|(dst, &src)| {
                        *dst = (src * multiplier).clamp(0.0, 255.0).round() as u8;
                    });
                Tensor::from_rvec_u8(core_types::RVec::from(u8_data), shape, TensorDType::U8)
                    .unwrap()
            } else {
                let mut chw_u8 = vec![0u8; dst_channels * height * width];
                let plane_size = height * width;
                chw_u8
                    .par_chunks_exact_mut(plane_size)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for y in 0..height {
                            for x in 0..width {
                                let src_val = hwc_f32[(y * width + x) * dst_channels + c];
                                plane[y * width + x] =
                                    (src_val * multiplier).clamp(0.0, 255.0).round() as u8;
                            }
                        }
                    });
                Tensor::from_rvec_u8(core_types::RVec::from(chw_u8), shape, TensorDType::U8)
                    .unwrap()
            }
        }
    }
}

fn standardize_raw_tensor(
    tensor: &Tensor,
    target_color: Option<ColorSpace>,
    target_dtype: TargetDType,
    target_layout: TargetLayout,
    normalize: Option<bool>,
) -> Tensor {
    let shape = tensor.shape.as_slice();
    if shape.len() < 2 || shape.len() > 3 {
        return tensor.clone();
    }

    let (_height, _width, channels, src_layout) = if shape.len() == 2 {
        (shape[0], shape[1], 1, ImageLayout::Hwc)
    } else if shape[2] <= 4 {
        (shape[0], shape[1], shape[2], ImageLayout::Hwc)
    } else if shape[0] <= 4 {
        (shape[1], shape[2], shape[0], ImageLayout::Chw)
    } else {
        (shape[0], shape[1], shape[2], ImageLayout::Hwc)
    };

    let color_space = match channels {
        1 => ColorSpace::Grayscale,
        3 => ColorSpace::Rgb,
        4 => ColorSpace::Rgba,
        _ => ColorSpace::Rgb,
    };

    if let Ok(img) = core_types::Image::new(tensor.clone(), color_space, src_layout) {
        standardize_image_to_tensor(&img, target_color, target_dtype, target_layout, normalize)
    } else {
        tensor.clone()
    }
}

#[inline]
fn extract_rgba_u8(src: &[u8], color: ColorSpace) -> (f32, f32, f32, f32) {
    match color {
        ColorSpace::Rgb => (src[0] as f32, src[1] as f32, src[2] as f32, 255.0),
        ColorSpace::Rgba => (src[0] as f32, src[1] as f32, src[2] as f32, src[3] as f32),
        ColorSpace::Bgr => (src[2] as f32, src[1] as f32, src[0] as f32, 255.0),
        ColorSpace::Bgra => (src[2] as f32, src[1] as f32, src[0] as f32, src[3] as f32),
        ColorSpace::Grayscale => (src[0] as f32, src[0] as f32, src[0] as f32, 255.0),
        ColorSpace::GrayscaleAlpha => (src[0] as f32, src[0] as f32, src[0] as f32, src[1] as f32),
        _ => (
            src[0] as f32,
            src.get(1).copied().unwrap_or(0) as f32,
            src.get(2).copied().unwrap_or(0) as f32,
            255.0,
        ),
    }
}

#[inline]
fn extract_rgba_u8_chw(
    src: &[u8],
    idx: usize,
    plane_size: usize,
    color: ColorSpace,
) -> (f32, f32, f32, f32) {
    match color {
        ColorSpace::Rgb => (
            src[idx] as f32,
            src[plane_size + idx] as f32,
            src[2 * plane_size + idx] as f32,
            255.0,
        ),
        ColorSpace::Rgba => (
            src[idx] as f32,
            src[plane_size + idx] as f32,
            src[2 * plane_size + idx] as f32,
            src[3 * plane_size + idx] as f32,
        ),
        ColorSpace::Bgr => (
            src[2 * plane_size + idx] as f32,
            src[plane_size + idx] as f32,
            src[idx] as f32,
            255.0,
        ),
        ColorSpace::Bgra => (
            src[2 * plane_size + idx] as f32,
            src[plane_size + idx] as f32,
            src[idx] as f32,
            src[3 * plane_size + idx] as f32,
        ),
        ColorSpace::Grayscale => (src[idx] as f32, src[idx] as f32, src[idx] as f32, 255.0),
        _ => (src[idx] as f32, src[idx] as f32, src[idx] as f32, 255.0),
    }
}

#[inline]
fn extract_rgba_f32(src: &[f32], color: ColorSpace) -> (f32, f32, f32, f32) {
    match color {
        ColorSpace::Rgb => (src[0], src[1], src[2], 1.0),
        ColorSpace::Rgba => (src[0], src[1], src[2], src[3]),
        ColorSpace::Bgr => (src[2], src[1], src[0], 1.0),
        ColorSpace::Bgra => (src[2], src[1], src[0], src[3]),
        ColorSpace::Grayscale => (src[0], src[0], src[0], 1.0),
        ColorSpace::GrayscaleAlpha => (src[0], src[0], src[0], src[1]),
        _ => (
            src[0],
            src.get(1).copied().unwrap_or(0.0),
            src.get(2).copied().unwrap_or(0.0),
            1.0,
        ),
    }
}

#[inline]
fn extract_rgba_f32_chw(
    src: &[f32],
    idx: usize,
    plane_size: usize,
    color: ColorSpace,
) -> (f32, f32, f32, f32) {
    match color {
        ColorSpace::Rgb => (
            src[idx],
            src[plane_size + idx],
            src[2 * plane_size + idx],
            1.0,
        ),
        ColorSpace::Rgba => (
            src[idx],
            src[plane_size + idx],
            src[2 * plane_size + idx],
            src[3 * plane_size + idx],
        ),
        ColorSpace::Bgr => (
            src[2 * plane_size + idx],
            src[plane_size + idx],
            src[idx],
            1.0,
        ),
        ColorSpace::Bgra => (
            src[2 * plane_size + idx],
            src[plane_size + idx],
            src[idx],
            src[3 * plane_size + idx],
        ),
        ColorSpace::Grayscale => (src[idx], src[idx], src[idx], 1.0),
        _ => (src[idx], src[idx], src[idx], 1.0),
    }
}

#[inline]
fn write_rgba_f32(dst: &mut [f32], color: ColorSpace, r: f32, g: f32, b: f32, a: f32) {
    match color {
        ColorSpace::Rgb => {
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }
        ColorSpace::Rgba => {
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
            dst[3] = a;
        }
        ColorSpace::Bgr => {
            dst[0] = b;
            dst[1] = g;
            dst[2] = r;
        }
        ColorSpace::Bgra => {
            dst[0] = b;
            dst[1] = g;
            dst[2] = r;
            dst[3] = a;
        }
        ColorSpace::Grayscale => {
            dst[0] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        ColorSpace::GrayscaleAlpha => {
            dst[0] = 0.299 * r + 0.587 * g + 0.114 * b;
            dst[1] = a;
        }
        _ => {
            if !dst.is_empty() {
                dst[0] = r;
            }
            if dst.len() > 1 {
                dst[1] = g;
            }
            if dst.len() > 2 {
                dst[2] = b;
            }
            if dst.len() > 3 {
                dst[3] = a;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Image, RBox, RString, Tuple2};

    #[test]
    fn test_to_tensor_u8_to_f32_rgb_hwc() {
        let u8_data = vec![255u8, 0, 128, 0, 255, 64];
        let img = Image::from_u8_hwc(&u8_data, 2, 1, ColorSpace::Rgb).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("dtype"), RString::from("f32")));
        named.push(Tuple2(RString::from("layout"), RString::from("hwc")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Image(img)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(t) = result {
            assert_eq!(t.shape.as_slice(), &[1, 2, 3]);
            assert_eq!(t.dtype, TensorDType::F32);
            let slice: &[f32] = t.as_f32_slice().unwrap();
            assert!((slice[0] - 1.0).abs() < 1e-4);
            assert_eq!(slice[1], 0.0);
            assert!((slice[2] - (128.0 / 255.0)).abs() < 1e-3);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }

    #[test]
    fn test_to_tensor_bgr_to_rgb_chw() {
        let u8_data = vec![10u8, 20, 30]; // B=10, G=20, R=30
        let img = Image::from_u8_hwc(&u8_data, 1, 1, ColorSpace::Bgr).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("color"), RString::from("rgb")));
        named.push(Tuple2(RString::from("layout"), RString::from("chw")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Image(img)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(t) = result {
            assert_eq!(t.shape.as_slice(), &[3, 1, 1]);
            let slice: &[f32] = t.as_f32_slice().unwrap();
            // R = 30, G = 20, B = 10
            assert!((slice[0] - (30.0 / 255.0)).abs() < 1e-3);
            assert!((slice[1] - (20.0 / 255.0)).abs() < 1e-3);
            assert!((slice[2] - (10.0 / 255.0)).abs() < 1e-3);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
