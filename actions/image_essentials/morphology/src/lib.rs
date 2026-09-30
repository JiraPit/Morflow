#![allow(clippy::too_many_arguments)]

use core_types::{
    ActionArgs, DataType, GetShapeResultFn, ImageLayout, Payload, Shape, ShapeResult, Tensor,
    TensorDType,
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

fn shape_impl(input: Shape, _args: ActionArgs) -> Shape {
    input
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeResultFn = get_output_shape_result;

#[no_mangle]
pub extern "C" fn get_output_shape_result(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args).into()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MorphOp {
    Dilate,
    Erode,
    Open,
    Close,
    Gradient,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MorphShape {
    Rect,
    Cross,
    Ellipse,
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut op = MorphOp::Dilate;
    let mut kernel_size = 3usize;
    let mut shape = MorphShape::Rect;
    let mut iterations = 1usize;

    if let Some(args) = args_opt {
        if let Some(op_str) = args
            .get_named("op")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            op = match op_str.to_lowercase().as_str() {
                "erode" | "erosion" => MorphOp::Erode,
                "open" | "opening" => MorphOp::Open,
                "close" | "closing" => MorphOp::Close,
                "gradient" | "grad" => MorphOp::Gradient,
                _ => MorphOp::Dilate,
            };
        }
        if let Some(k_str) = args
            .get_named("kernel_size")
            .or_else(|| args.get_named("ksize"))
        {
            if let Ok(k) = k_str.parse::<usize>() {
                kernel_size = k.max(1);
            }
        }
        if let Some(s_str) = args.get_named("shape") {
            shape = match s_str.to_lowercase().as_str() {
                "cross" => MorphShape::Cross,
                "ellipse" | "circle" => MorphShape::Ellipse,
                _ => MorphShape::Rect,
            };
        }
        if let Some(it_str) = args
            .get_named("iterations")
            .or_else(|| args.get_named("iter"))
        {
            if let Ok(it) = it_str.parse::<usize>() {
                iterations = it.max(1);
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
            let res = apply_morphology(&tensor, layout, op, kernel_size, shape, iterations);
            Payload::Tensor(res)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'morphology\' requires Payload::Tensor",
        )),
    }
}

fn apply_morphology(
    tensor: &Tensor,
    layout: ImageLayout,
    op: MorphOp,
    ksize: usize,
    shape: MorphShape,
    iterations: usize,
) -> Tensor {
    let tensor_shape = tensor.shape.as_slice();
    let (height, width, channels) = match (tensor_shape.len(), layout) {
        (2, _) => (tensor_shape[0], tensor_shape[1], 1),
        (3, ImageLayout::Hwc) => (tensor_shape[0], tensor_shape[1], tensor_shape[2]),
        (3, ImageLayout::Chw) => (tensor_shape[1], tensor_shape[2], tensor_shape[0]),
        _ => return tensor.clone(),
    };

    if height == 0 || width == 0 {
        return tensor.clone();
    }

    let radius = ksize / 2;
    if radius == 0 {
        return tensor.clone();
    }

    match tensor.dtype {
        TensorDType::F32 => {
            let Some(src) = tensor.as_f32_slice() else {
                return tensor.clone();
            };

            let out_f32 =
                apply_morph_ops(src, width, height, channels, radius, shape, op, iterations);

            let out_shape = if channels == 1 {
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

            let out_f32 = apply_morph_ops(
                &src_f32, width, height, channels, radius, shape, op, iterations,
            );

            let mut out_u8 = vec![0u8; out_f32.len()];
            out_u8
                .par_iter_mut()
                .zip(out_f32.par_iter())
                .for_each(|(dst, &f)| {
                    *dst = (f * 255.0).clamp(0.0, 255.0).round() as u8;
                });

            let out_shape = if channels == 1 {
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

fn apply_morph_ops(
    src: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    radius: usize,
    shape: MorphShape,
    op: MorphOp,
    iterations: usize,
) -> Vec<f32> {
    match op {
        MorphOp::Dilate => {
            let mut cur = src.to_vec();
            for _ in 0..iterations {
                cur = dilate_f32(&cur, width, height, channels, radius, shape);
            }
            cur
        }
        MorphOp::Erode => {
            let mut cur = src.to_vec();
            for _ in 0..iterations {
                cur = erode_f32(&cur, width, height, channels, radius, shape);
            }
            cur
        }
        MorphOp::Open => {
            let mut cur = src.to_vec();
            for _ in 0..iterations {
                cur = erode_f32(&cur, width, height, channels, radius, shape);
            }
            for _ in 0..iterations {
                cur = dilate_f32(&cur, width, height, channels, radius, shape);
            }
            cur
        }
        MorphOp::Close => {
            let mut cur = src.to_vec();
            for _ in 0..iterations {
                cur = dilate_f32(&cur, width, height, channels, radius, shape);
            }
            for _ in 0..iterations {
                cur = erode_f32(&cur, width, height, channels, radius, shape);
            }
            cur
        }
        MorphOp::Gradient => {
            let dilated = dilate_f32(src, width, height, channels, radius, shape);
            let eroded = erode_f32(src, width, height, channels, radius, shape);
            let mut grad = vec![0.0f32; src.len()];
            grad.par_iter_mut()
                .zip(dilated.par_iter())
                .zip(eroded.par_iter())
                .for_each(|((dst, &d), &e)| {
                    *dst = (d - e).max(0.0);
                });
            grad
        }
    }
}

fn dilate_f32(
    src: &[f32],
    w: usize,
    h: usize,
    c: usize,
    radius: usize,
    shape: MorphShape,
) -> Vec<f32> {
    if shape == MorphShape::Rect {
        // Separable 1D horizontal max + 1D vertical max
        let mut temp = vec![0.0f32; h * w * c];
        let mut out = vec![0.0f32; h * w * c];

        temp.par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                let src_row = &src[y * w * c..(y + 1) * w * c];
                for x in 0..w {
                    for ch in 0..c {
                        let mut max_val = f32::NEG_INFINITY;
                        for k in -(radius as isize)..=(radius as isize) {
                            let sx = (x as isize + k).clamp(0, w as isize - 1) as usize;
                            let v = src_row[sx * c + ch];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        row[x * c + ch] = max_val;
                    }
                }
            });

        out.par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    for ch in 0..c {
                        let mut max_val = f32::NEG_INFINITY;
                        for k in -(radius as isize)..=(radius as isize) {
                            let sy = (y as isize + k).clamp(0, h as isize - 1) as usize;
                            let v = temp[(sy * w + x) * c + ch];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        row[x * c + ch] = max_val;
                    }
                }
            });

        out
    } else {
        let mut out = vec![0.0f32; h * w * c];
        out.par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    for ch in 0..c {
                        let mut max_val = f32::NEG_INFINITY;
                        for dy in -(radius as isize)..=(radius as isize) {
                            let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                            for dx in -(radius as isize)..=(radius as isize) {
                                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                                let inside = match shape {
                                    MorphShape::Cross => dx == 0 || dy == 0,
                                    MorphShape::Ellipse => {
                                        (dx * dx + dy * dy) as usize <= radius * radius
                                    }
                                    _ => true,
                                };
                                if inside {
                                    let v = src[(sy * w + sx) * c + ch];
                                    if v > max_val {
                                        max_val = v;
                                    }
                                }
                            }
                        }
                        row[x * c + ch] = max_val;
                    }
                }
            });
        out
    }
}

fn erode_f32(
    src: &[f32],
    w: usize,
    h: usize,
    c: usize,
    radius: usize,
    shape: MorphShape,
) -> Vec<f32> {
    if shape == MorphShape::Rect {
        // Separable 1D horizontal min + 1D vertical min
        let mut temp = vec![0.0f32; h * w * c];
        let mut out = vec![0.0f32; h * w * c];

        temp.par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                let src_row = &src[y * w * c..(y + 1) * w * c];
                for x in 0..w {
                    for ch in 0..c {
                        let mut min_val = f32::INFINITY;
                        for k in -(radius as isize)..=(radius as isize) {
                            let sx = (x as isize + k).clamp(0, w as isize - 1) as usize;
                            let v = src_row[sx * c + ch];
                            if v < min_val {
                                min_val = v;
                            }
                        }
                        row[x * c + ch] = min_val;
                    }
                }
            });

        out.par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    for ch in 0..c {
                        let mut min_val = f32::INFINITY;
                        for k in -(radius as isize)..=(radius as isize) {
                            let sy = (y as isize + k).clamp(0, h as isize - 1) as usize;
                            let v = temp[(sy * w + x) * c + ch];
                            if v < min_val {
                                min_val = v;
                            }
                        }
                        row[x * c + ch] = min_val;
                    }
                }
            });

        out
    } else {
        let mut out = vec![0.0f32; h * w * c];
        out.par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..w {
                    for ch in 0..c {
                        let mut min_val = f32::INFINITY;
                        for dy in -(radius as isize)..=(radius as isize) {
                            let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                            for dx in -(radius as isize)..=(radius as isize) {
                                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                                let inside = match shape {
                                    MorphShape::Cross => dx == 0 || dy == 0,
                                    MorphShape::Ellipse => {
                                        (dx * dx + dy * dy) as usize <= radius * radius
                                    }
                                    _ => true,
                                };
                                if inside {
                                    let v = src[(sy * w + sx) * c + ch];
                                    if v < min_val {
                                        min_val = v;
                                    }
                                }
                            }
                        }
                        row[x * c + ch] = min_val;
                    }
                }
            });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_morphology_dilation() {
        let f32_data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![3, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("op"), RString::from("dilate")));
        named.push(Tuple2(RString::from("kernel_size"), RString::from("3")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            let slice: &[f32] = out_t.as_f32_slice().unwrap();
            // A 1 at center with 3x3 rect dilation expands to all 1s
            for &val in slice {
                assert_eq!(val, 1.0);
            }
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
