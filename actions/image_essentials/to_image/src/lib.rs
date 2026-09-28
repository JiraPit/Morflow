use core_types::{ColorSpace, DataType, Image, ImageLayout, Payload, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Image
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut target_color: Option<ColorSpace> = None;
    let mut target_dtype = TensorDType::U8;
    let mut target_layout = ImageLayout::Hwc;
    let mut denormalize: Option<bool> = None;

    if let Some(args) = args_opt {
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
                "f32" | "float" | "float32" => target_dtype = TensorDType::F32,
                "u8" | "uint8" | "byte" | "bytes" => target_dtype = TensorDType::U8,
                _ => {}
            }
        }
        if let Some(lay) = args.get_named("layout") {
            match lay.to_lowercase().as_str() {
                "chw" | "planar" => target_layout = ImageLayout::Chw,
                "hwc" | "interleaved" => target_layout = ImageLayout::Hwc,
                _ => {}
            }
        }
        if let Some(denorm_str) = args
            .get_named("denormalize")
            .or_else(|| args.get_named("unnormalize"))
        {
            denormalize = match denorm_str.to_lowercase().as_str() {
                "true" | "1" | "yes" => Some(true),
                "false" | "0" | "no" => Some(false),
                _ => None,
            };
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => {
            let img = tensor_to_image(
                &tensor,
                target_color,
                target_dtype,
                target_layout,
                denormalize,
            );
            Payload::Image(img)
        }
        Payload::Image(img) => {
            let converted = tensor_to_image(
                &img.tensor,
                target_color.or(Some(img.color_space)),
                target_dtype,
                target_layout,
                denormalize,
            );
            Payload::Image(converted)
        }
        other => other,
    }
}

fn tensor_to_image(
    tensor: &Tensor,
    target_color: Option<ColorSpace>,
    target_dtype: TensorDType,
    target_layout: ImageLayout,
    denormalize: Option<bool>,
) -> Image {
    let shape = tensor.shape.as_slice();
    let should_denorm = denormalize.unwrap_or(true);

    let (height, width, in_channels, in_layout) = if shape.len() == 2 {
        (shape[0], shape[1], 1, ImageLayout::Hwc)
    } else if shape.len() == 3 {
        if shape[2] <= 4 {
            (shape[0], shape[1], shape[2], ImageLayout::Hwc)
        } else if shape[0] <= 4 {
            (shape[1], shape[2], shape[0], ImageLayout::Chw)
        } else {
            (shape[0], shape[1], shape[2], ImageLayout::Hwc)
        }
    } else {
        (1, tensor.num_elements(), 1, ImageLayout::Hwc)
    };

    let color_space = target_color.unwrap_or_else(|| match in_channels {
        1 => ColorSpace::Grayscale,
        3 => ColorSpace::Rgb,
        4 => ColorSpace::Rgba,
        _ => ColorSpace::Rgb,
    });

    let out_channels = color_space.channels();

    // Standardize input tensor to F32 HWC intermediate
    let hwc_f32: Vec<f32> = match tensor.dtype {
        TensorDType::F32 => {
            let Some(src_f32) = tensor.as_f32_slice() else {
                return Image {
                    tensor: tensor.clone(),
                    color_space,
                    layout: target_layout,
                };
            };
            if in_layout == ImageLayout::Hwc {
                src_f32.to_vec()
            } else {
                let mut hwc = vec![0.0f32; height * width * in_channels];
                let plane_size = height * width;
                hwc.par_chunks_exact_mut(width * in_channels)
                    .enumerate()
                    .for_each(|(y, row)| {
                        for x in 0..width {
                            for c in 0..in_channels {
                                row[x * in_channels + c] = src_f32[c * plane_size + y * width + x];
                            }
                        }
                    });
                hwc
            }
        }
        TensorDType::U8 => {
            let Some(bytes) = tensor.as_u8_slice() else {
                return Image {
                    tensor: tensor.clone(),
                    color_space,
                    layout: target_layout,
                };
            };
            let mut hwc = vec![0.0f32; height * width * in_channels];
            if in_layout == ImageLayout::Hwc {
                hwc.par_iter_mut()
                    .zip(bytes.par_iter())
                    .for_each(|(dst, &src)| {
                        *dst = src as f32 / 255.0;
                    });
            } else {
                let plane_size = height * width;
                hwc.par_chunks_exact_mut(width * in_channels)
                    .enumerate()
                    .for_each(|(y, row)| {
                        for x in 0..width {
                            for c in 0..in_channels {
                                row[x * in_channels + c] =
                                    bytes[c * plane_size + y * width + x] as f32 / 255.0;
                            }
                        }
                    });
            }
            hwc
        }
        _ => {
            return Image::new(tensor.clone(), color_space, target_layout).unwrap_or_else(|_| {
                let dummy =
                    Tensor::from_rvec_u8(core_types::RVec::new(), vec![0, 0], TensorDType::U8)
                        .unwrap();
                Image {
                    tensor: dummy,
                    color_space,
                    layout: target_layout,
                }
            });
        }
    };

    // Adapt channels if in_channels != out_channels
    let final_hwc_f32: Vec<f32> = if in_channels == out_channels {
        hwc_f32
    } else {
        let mut adapted = vec![0.0f32; height * width * out_channels];
        adapted
            .par_chunks_exact_mut(width * out_channels)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let in_idx = (y * width + x) * in_channels;
                    let out_idx = x * out_channels;
                    match (in_channels, out_channels) {
                        (1, 3) => {
                            let v = hwc_f32[in_idx];
                            row[out_idx] = v;
                            row[out_idx + 1] = v;
                            row[out_idx + 2] = v;
                        }
                        (1, 4) => {
                            let v = hwc_f32[in_idx];
                            row[out_idx] = v;
                            row[out_idx + 1] = v;
                            row[out_idx + 2] = v;
                            row[out_idx + 3] = 1.0;
                        }
                        (3, 4) => {
                            row[out_idx] = hwc_f32[in_idx];
                            row[out_idx + 1] = hwc_f32[in_idx + 1];
                            row[out_idx + 2] = hwc_f32[in_idx + 2];
                            row[out_idx + 3] = 1.0;
                        }
                        (4, 3) => {
                            row[out_idx] = hwc_f32[in_idx];
                            row[out_idx + 1] = hwc_f32[in_idx + 1];
                            row[out_idx + 2] = hwc_f32[in_idx + 2];
                        }
                        (3, 1) | (4, 1) => {
                            let r = hwc_f32[in_idx];
                            let g = hwc_f32[in_idx + 1];
                            let b = hwc_f32[in_idx + 2];
                            row[out_idx] = 0.299 * r + 0.587 * g + 0.114 * b;
                        }
                        _ => {
                            for c in 0..out_channels {
                                row[out_idx + c] = if c < in_channels {
                                    hwc_f32[in_idx + c]
                                } else {
                                    1.0
                                };
                            }
                        }
                    }
                }
            });
        adapted
    };

    // Output layout and dtype packaging
    let out_tensor = match (target_dtype, target_layout) {
        (TensorDType::U8, ImageLayout::Hwc) => {
            let shape = if out_channels == 1 {
                vec![height, width]
            } else {
                vec![height, width, out_channels]
            };
            let mut u8_data = vec![0u8; height * width * out_channels];
            let multiplier = if should_denorm { 255.0 } else { 1.0 };
            u8_data
                .par_iter_mut()
                .zip(final_hwc_f32.par_iter())
                .for_each(|(dst, &src)| {
                    *dst = (src * multiplier).clamp(0.0, 255.0).round() as u8;
                });
            Tensor::from_rvec_u8(core_types::RVec::from(u8_data), shape, TensorDType::U8).unwrap()
        }
        (TensorDType::U8, ImageLayout::Chw) => {
            let shape = if out_channels == 1 {
                vec![height, width]
            } else {
                vec![out_channels, height, width]
            };
            let multiplier = if should_denorm { 255.0 } else { 1.0 };
            if out_channels == 1 {
                let mut u8_data = vec![0u8; height * width];
                u8_data
                    .par_iter_mut()
                    .zip(final_hwc_f32.par_iter())
                    .for_each(|(dst, &src)| {
                        *dst = (src * multiplier).clamp(0.0, 255.0).round() as u8;
                    });
                Tensor::from_rvec_u8(core_types::RVec::from(u8_data), shape, TensorDType::U8)
                    .unwrap()
            } else {
                let mut chw_u8 = vec![0u8; out_channels * height * width];
                let plane_size = height * width;
                chw_u8
                    .par_chunks_exact_mut(plane_size)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for y in 0..height {
                            for x in 0..width {
                                let src_val = final_hwc_f32[(y * width + x) * out_channels + c];
                                plane[y * width + x] =
                                    (src_val * multiplier).clamp(0.0, 255.0).round() as u8;
                            }
                        }
                    });
                Tensor::from_rvec_u8(core_types::RVec::from(chw_u8), shape, TensorDType::U8)
                    .unwrap()
            }
        }
        (TensorDType::F32, ImageLayout::Hwc) => {
            let shape = if out_channels == 1 {
                vec![height, width]
            } else {
                vec![height, width, out_channels]
            };
            Tensor::from_f32_shape(&final_hwc_f32, shape).unwrap()
        }
        (TensorDType::F32, ImageLayout::Chw) => {
            let shape = if out_channels == 1 {
                vec![height, width]
            } else {
                vec![out_channels, height, width]
            };
            if out_channels == 1 {
                Tensor::from_f32_shape(&final_hwc_f32, shape).unwrap()
            } else {
                let mut chw = vec![0.0f32; out_channels * height * width];
                let plane_size = height * width;
                chw.par_chunks_exact_mut(plane_size)
                    .enumerate()
                    .for_each(|(c, plane)| {
                        for y in 0..height {
                            for x in 0..width {
                                plane[y * width + x] =
                                    final_hwc_f32[(y * width + x) * out_channels + c];
                            }
                        }
                    });
                Tensor::from_f32_shape(&chw, shape).unwrap()
            }
        }
        _ => tensor.clone(),
    };

    Image {
        tensor: out_tensor,
        color_space,
        layout: target_layout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_to_image_from_tensor_f32() {
        let f32_data = vec![1.0f32, 0.0, 0.5, 0.0, 1.0, 0.25];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![1, 2, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("color"), RString::from("rgb")));
        named.push(Tuple2(RString::from("dtype"), RString::from("u8")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Image(img) = result {
            assert_eq!(img.width(), 2);
            assert_eq!(img.height(), 1);
            assert_eq!(img.channels(), 3);
            assert_eq!(img.color_space, ColorSpace::Rgb);
            assert_eq!(img.dtype(), TensorDType::U8);

            let bytes = img.as_u8_slice().unwrap();
            assert_eq!(bytes[0], 255);
            assert_eq!(bytes[1], 0);
            assert_eq!(bytes[2], 128);
        } else {
            panic!("Expected Payload::Image");
        }
    }
}
