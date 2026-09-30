use core_types::{
    ActionArgs, DataType, GetShapeFn, ImageLayout, Payload, Shape, ShapeResult, Tensor, TensorDType,
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

fn shape_impl(input: Shape, _args: ActionArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if !matches!(input.rank(), 2 | 3) {
            return Err("Image operations require rank 2 or 3".into());
        }
        Ok(input)
    })())
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeMode {
    Sobel,
    SobelX,
    SobelY,
    Laplacian,
    Prewitt,
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let payload = match core_types::contract::image_input(payload, true) {
        Ok(payload) => payload,
        Err(error) => return Payload::Error(error),
    };
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut mode = EdgeMode::Sobel;
    let mut strength = 1.0f32;

    if let Some(args) = args_opt {
        if let Some(m_str) = args
            .get_named("mode")
            .or_else(|| args.get_named("filter"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            mode = match m_str.to_lowercase().as_str() {
                "sobel_x" | "sobelx" | "dx" => EdgeMode::SobelX,
                "sobel_y" | "sobely" | "dy" => EdgeMode::SobelY,
                "laplacian" | "laplace" => EdgeMode::Laplacian,
                "prewitt" => EdgeMode::Prewitt,
                _ => EdgeMode::Sobel,
            };
        }
        if let Some(st_str) = args
            .get_named("strength")
            .or_else(|| args.get_named("scale"))
        {
            if let Ok(st) = st_str.parse::<f32>() {
                strength = st;
            }
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
            let res = apply_edge_detect(&tensor, layout, mode, strength);
            Payload::Tensor(res)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'edge_detect\' requires Payload::Tensor",
        )),
    }
}

fn apply_edge_detect(
    tensor: &Tensor,
    layout: ImageLayout,
    mode: EdgeMode,
    strength: f32,
) -> Tensor {
    let shape = tensor.shape.as_slice();
    let (height, width, channels) = match (shape.len(), layout) {
        (2, _) => (shape[0], shape[1], 1),
        (3, ImageLayout::Hwc) => (shape[0], shape[1], shape[2]),
        (3, ImageLayout::Chw) => (shape[1], shape[2], shape[0]),
        _ => return tensor.clone(),
    };

    if height == 0 || width == 0 {
        return tensor.clone();
    }

    match tensor.dtype {
        TensorDType::F32 => {
            let Some(src) = tensor.as_f32_slice() else {
                return tensor.clone();
            };

            let out_f32 = match mode {
                EdgeMode::Sobel => convolve_sobel_mag_f32(src, width, height, channels, strength),
                EdgeMode::SobelX => convolve_sobel_x_f32(src, width, height, channels, strength),
                EdgeMode::SobelY => convolve_sobel_y_f32(src, width, height, channels, strength),
                EdgeMode::Laplacian => {
                    convolve_laplacian_f32(src, width, height, channels, strength)
                }
                EdgeMode::Prewitt => {
                    convolve_prewitt_mag_f32(src, width, height, channels, strength)
                }
            };

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![height, width]
            } else if layout == ImageLayout::Hwc {
                vec![height, width, channels]
            } else {
                vec![channels, height, width]
            };
            Tensor::from_f32_vec(out_f32, out_shape).unwrap()
        }
        TensorDType::U8 => {
            let Some(bytes) = tensor.as_u8_slice() else {
                return tensor.clone();
            };
            let mut src_f32 = vec![0.0f32; bytes.len()];
            src_f32
                .par_iter_mut()
                .zip(bytes.par_iter())
                .for_each(|(dst, &b)| *dst = b as f32 / 255.0);

            let out_f32 = match mode {
                EdgeMode::Sobel => {
                    convolve_sobel_mag_f32(&src_f32, width, height, channels, strength)
                }
                EdgeMode::SobelX => {
                    convolve_sobel_x_f32(&src_f32, width, height, channels, strength)
                }
                EdgeMode::SobelY => {
                    convolve_sobel_y_f32(&src_f32, width, height, channels, strength)
                }
                EdgeMode::Laplacian => {
                    convolve_laplacian_f32(&src_f32, width, height, channels, strength)
                }
                EdgeMode::Prewitt => {
                    convolve_prewitt_mag_f32(&src_f32, width, height, channels, strength)
                }
            };

            let mut out_u8 = vec![0u8; out_f32.len()];
            out_u8
                .par_iter_mut()
                .zip(out_f32.par_iter())
                .for_each(|(dst, &f)| {
                    *dst = (f * 255.0).clamp(0.0, 255.0).round() as u8;
                });

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![height, width]
            } else if layout == ImageLayout::Hwc {
                vec![height, width, channels]
            } else {
                vec![channels, height, width]
            };
            Tensor::from_rvec_u8(core_types::RVec::from(out_u8), out_shape, TensorDType::U8)
                .unwrap()
        }
        _ => tensor.clone(),
    }
}

fn convolve_sobel_mag_f32(src: &[f32], w: usize, h: usize, c: usize, strength: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c];
    out.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                for ch in 0..c {
                    let mut gx = 0.0f32;
                    let mut gy = 0.0f32;
                    for dy in -1..=1 {
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        let weight_y = if dy == 0 { 2.0 } else { 1.0 };
                        let sy_weight = dy as f32;
                        for dx in -1..=1 {
                            let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                            let weight_x = if dx == 0 { 2.0 } else { 1.0 };
                            let sx_weight = dx as f32;
                            let val = src[(sy * w + sx) * c + ch];
                            gx += weight_y * sx_weight * val;
                            gy += sy_weight * weight_x * val;
                        }
                    }
                    row[x * c + ch] = (gx * gx + gy * gy).sqrt() * strength;
                }
            }
        });
    out
}

fn convolve_sobel_x_f32(src: &[f32], w: usize, h: usize, c: usize, strength: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c];
    out.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for dy in -1..=1 {
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        let weight_y = if dy == 0 { 2.0 } else { 1.0 };
                        for dx in -1..=1 {
                            let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                            let weight_x = dx as f32;
                            sum += weight_y * weight_x * src[(sy * w + sx) * c + ch];
                        }
                    }
                    row[x * c + ch] = (sum * strength).abs();
                }
            }
        });
    out
}

fn convolve_sobel_y_f32(src: &[f32], w: usize, h: usize, c: usize, strength: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c];
    out.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for dy in -1..=1 {
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        let weight_y = dy as f32;
                        for dx in -1..=1 {
                            let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                            let weight_x = if dx == 0 { 2.0 } else { 1.0 };
                            sum += weight_y * weight_x * src[(sy * w + sx) * c + ch];
                        }
                    }
                    row[x * c + ch] = (sum * strength).abs();
                }
            }
        });
    out
}

fn convolve_laplacian_f32(src: &[f32], w: usize, h: usize, c: usize, strength: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c];
    out.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                for ch in 0..c {
                    let center = src[(y * w + x) * c + ch];
                    let up =
                        src[((y as isize - 1).clamp(0, h as isize - 1) as usize * w + x) * c + ch];
                    let down =
                        src[((y as isize + 1).clamp(0, h as isize - 1) as usize * w + x) * c + ch];
                    let left =
                        src[(y * w + (x as isize - 1).clamp(0, w as isize - 1) as usize) * c + ch];
                    let right =
                        src[(y * w + (x as isize + 1).clamp(0, w as isize - 1) as usize) * c + ch];

                    let lap = up + down + left + right - 4.0 * center;
                    row[x * c + ch] = (lap * strength).abs();
                }
            }
        });
    out
}

fn convolve_prewitt_mag_f32(src: &[f32], w: usize, h: usize, c: usize, strength: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w * c];
    out.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                for ch in 0..c {
                    let mut gx = 0.0f32;
                    let mut gy = 0.0f32;
                    for dy in -1..=1 {
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        for dx in -1..=1 {
                            let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                            let val = src[(sy * w + sx) * c + ch];
                            gx += dx as f32 * val;
                            gy += dy as f32 * val;
                        }
                    }
                    row[x * c + ch] = (gx * gx + gy * gy).sqrt() * strength;
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_edge_detect_sobel() {
        let f32_data = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![4, 4]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("mode"), RString::from("sobel_x")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[4, 4]);
            let slice: &[f32] = out_t.as_f32_slice().unwrap();
            // Vertical step between col 1 (0.0) and col 2 (1.0) produces strong gradient
            assert!(slice[5] > 0.0);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
