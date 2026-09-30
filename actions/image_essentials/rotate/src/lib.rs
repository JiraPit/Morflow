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

fn shape_impl(input: Shape, args: ActionArgs) -> ShapeResult {
    use core_types::contract::{self, arg, Error};
    contract::finish((|| {
        let (h, w, c, _chw) = contract::image_dims(&input)?;
        let angle = arg::<f32>(&args, &["angle", "angle_deg"], Some(0), Some(90.0))?.unwrap();
        if !angle.is_finite() {
            return Err("Rotation angle must be finite".into());
        }
        let expand = contract::value(&args, &["expand", "expand_canvas"], None)?
            .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
            .unwrap_or(true);
        if h == 0 || w == 0 {
            return Err(Error::Unknown);
        }
        let norm = ((angle % 360.0) + 360.0) % 360.0;
        let (oh, ow) = if (norm - 90.0).abs() < 1e-3 || (norm - 270.0).abs() < 1e-3 {
            (w, h)
        } else if norm.abs() < 1e-3 || (norm - 180.0).abs() < 1e-3 || !expand {
            (h, w)
        } else {
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
            (oh as usize, ow as usize)
        };
        contract::image_shape(oh, ow, c, input.rank(), false)
    })())
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
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
    let mut angle_deg = 90.0f32;
    let mut expand_canvas = true;
    let mut fill_value = 0.0f32;

    if let Some(args) = args_opt {
        if let Some(a_str) = args
            .get_named("angle")
            .or_else(|| args.get_named("angle_deg"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(a) = a_str.parse::<f32>() {
                angle_deg = a;
            }
        }
        if let Some(exp_str) = args
            .get_named("expand")
            .or_else(|| args.get_named("expand_canvas"))
        {
            expand_canvas = exp_str.to_lowercase() == "true" || exp_str == "1";
        }
        if let Some(fill_str) = args
            .get_named("fill")
            .or_else(|| args.get_named("fill_value"))
        {
            if let Ok(f) = fill_str.parse::<f32>() {
                fill_value = f;
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
            let res = apply_rotate(&tensor, layout, angle_deg, expand_canvas, fill_value);
            Payload::Tensor(res)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'rotate\' requires Payload::Tensor",
        )),
    }
}

fn apply_rotate(
    tensor: &Tensor,
    layout: ImageLayout,
    angle_deg: f32,
    expand_canvas: bool,
    fill_value: f32,
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

    let norm_angle = ((angle_deg % 360.0) + 360.0) % 360.0;

    match tensor.dtype {
        TensorDType::F32 => {
            let Some(src) = tensor.as_f32_slice() else {
                return tensor.clone();
            };

            let (out_w, out_h, out_data) = rotate_f32_buffer(
                src,
                in_w,
                in_h,
                channels,
                norm_angle,
                expand_canvas,
                fill_value,
            );

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![out_h, out_w]
            } else {
                vec![out_h, out_w, channels]
            };
            Tensor::from_f32_vec(out_data, out_shape).unwrap()
        }
        TensorDType::U8 => {
            let Some(bytes) = tensor.as_u8_slice() else {
                return tensor.clone();
            };
            let mut f32_src = vec![0.0f32; bytes.len()];
            f32_src
                .par_iter_mut()
                .zip(bytes.par_iter())
                .for_each(|(dst, &b)| *dst = b as f32 / 255.0);

            let (out_w, out_h, out_data) = rotate_f32_buffer(
                &f32_src,
                in_w,
                in_h,
                channels,
                norm_angle,
                expand_canvas,
                fill_value / 255.0,
            );

            let mut out_u8 = vec![0u8; out_data.len()];
            out_u8
                .par_iter_mut()
                .zip(out_data.par_iter())
                .for_each(|(dst, &f)| {
                    *dst = (f * 255.0).clamp(0.0, 255.0).round() as u8;
                });

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![out_h, out_w]
            } else {
                vec![out_h, out_w, channels]
            };
            Tensor::from_rvec_u8(core_types::RVec::from(out_u8), out_shape, TensorDType::U8)
                .unwrap()
        }
        _ => tensor.clone(),
    }
}

fn rotate_f32_buffer(
    src: &[f32],
    in_w: usize,
    in_h: usize,
    channels: usize,
    norm_angle: f32,
    expand_canvas: bool,
    fill_value: f32,
) -> (usize, usize, Vec<f32>) {
    if (norm_angle - 0.0).abs() < 1e-3 {
        (in_w, in_h, src.to_vec())
    } else if (norm_angle - 90.0).abs() < 1e-3 {
        // 90 deg CW
        let out_w = in_h;
        let out_h = in_w;
        let mut out = vec![0.0f32; out_h * out_w * channels];
        out.par_chunks_exact_mut(out_w * channels)
            .enumerate()
            .for_each(|(out_y, row)| {
                for out_x in 0..out_w {
                    let in_x = out_y;
                    let in_y = in_h - 1 - out_x;
                    let src_idx = (in_y * in_w + in_x) * channels;
                    let dst_idx = out_x * channels;
                    row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&src[src_idx..src_idx + channels]);
                }
            });
        (out_w, out_h, out)
    } else if (norm_angle - 180.0).abs() < 1e-3 {
        let mut out = vec![0.0f32; in_h * in_w * channels];
        out.par_chunks_exact_mut(in_w * channels)
            .enumerate()
            .for_each(|(y, row)| {
                let sy = in_h - 1 - y;
                for x in 0..in_w {
                    let sx = in_w - 1 - x;
                    let src_idx = (sy * in_w + sx) * channels;
                    let dst_idx = x * channels;
                    row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&src[src_idx..src_idx + channels]);
                }
            });
        (in_w, in_h, out)
    } else if (norm_angle - 270.0).abs() < 1e-3 {
        // 270 deg CW
        let out_w = in_h;
        let out_h = in_w;
        let mut out = vec![0.0f32; out_h * out_w * channels];
        out.par_chunks_exact_mut(out_w * channels)
            .enumerate()
            .for_each(|(out_y, row)| {
                for out_x in 0..out_w {
                    let in_x = in_w - 1 - out_y;
                    let in_y = out_x;
                    let src_idx = (in_y * in_w + in_x) * channels;
                    let dst_idx = out_x * channels;
                    row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&src[src_idx..src_idx + channels]);
                }
            });
        (out_w, out_h, out)
    } else {
        // Arbitrary angle rotation via 2D inverse affine
        let rad = norm_angle.to_radians();
        let cos_a = rad.cos();
        let sin_a = rad.sin();

        let (out_w, out_h) = if expand_canvas {
            let w_f = in_w as f32;
            let h_f = in_h as f32;
            let new_w = (w_f * cos_a.abs() + h_f * sin_a.abs()).round() as usize;
            let new_h = (w_f * sin_a.abs() + h_f * cos_a.abs()).round() as usize;
            (new_w.max(1), new_h.max(1))
        } else {
            (in_w, in_h)
        };

        let in_cx = in_w as f32 / 2.0;
        let in_cy = in_h as f32 / 2.0;
        let out_cx = out_w as f32 / 2.0;
        let out_cy = out_h as f32 / 2.0;

        let mut out = vec![fill_value; out_h * out_w * channels];
        out.par_chunks_exact_mut(out_w * channels)
            .enumerate()
            .for_each(|(out_y, row)| {
                let dy = out_y as f32 - out_cy;
                for out_x in 0..out_w {
                    let dx = out_x as f32 - out_cx;
                    let in_x_f = dx * cos_a + dy * sin_a + in_cx;
                    let in_y_f = -dx * sin_a + dy * cos_a + in_cy;
                    let dst_idx = out_x * channels;

                    if in_x_f >= 0.0
                        && in_x_f < (in_w - 1) as f32
                        && in_y_f >= 0.0
                        && in_y_f < (in_h - 1) as f32
                    {
                        let x0 = in_x_f.floor() as usize;
                        let y0 = in_y_f.floor() as usize;
                        let x1 = x0 + 1;
                        let y1 = y0 + 1;

                        let fx = in_x_f - x0 as f32;
                        let fy = in_y_f - y0 as f32;

                        let w00 = (1.0 - fx) * (1.0 - fy);
                        let w10 = fx * (1.0 - fy);
                        let w01 = (1.0 - fx) * fy;
                        let w11 = fx * fy;

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
                    } else {
                        for c in 0..channels {
                            row[dst_idx + c] = fill_value;
                        }
                    }
                }
            });

        (out_w, out_h, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_rotate_90() {
        let f32_data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![2, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("angle"), RString::from("90")));

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
            let slice: &[f32] = out_t.as_f32_slice().unwrap();
            assert_eq!(slice, &[3.0, 1.0, 4.0, 2.0]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
