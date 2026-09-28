use core_types::{DataType, Image, ImageLayout, Payload, Tensor, TensorDType};
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
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut angle_deg = 90.0f32;
    let mut expand_canvas = true;
    let mut fill_value = 0.0f32;

    if let Some(args) = payload.args() {
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

    match payload.unwrap_payload() {
        Payload::Image(img) => {
            let res = apply_rotate(
                &img.tensor,
                img.layout,
                angle_deg,
                expand_canvas,
                fill_value,
            );
            Payload::Image(Image {
                tensor: res,
                color_space: img.color_space,
                layout: img.layout,
            })
        }
        Payload::Tensor(tensor) => {
            let layout = if tensor.shape.len() == 3 && tensor.shape[2] <= 4 {
                ImageLayout::Hwc
            } else if tensor.shape.len() == 3 && tensor.shape[0] <= 4 {
                ImageLayout::Chw
            } else {
                ImageLayout::Hwc
            };
            let res = apply_rotate(tensor, layout, angle_deg, expand_canvas, fill_value);
            Payload::Tensor(res)
        }
        other => other.clone(),
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
            let bytes = tensor.to_contiguous_bytes();
            let src: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };

            let (out_w, out_h, out_data) = if (norm_angle - 0.0).abs() < 1e-3 {
                return tensor.clone();
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
                            for c in 0..channels {
                                row[dst_idx + c] = src[src_idx + c];
                            }
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
                            for c in 0..channels {
                                row[dst_idx + c] = src[src_idx + c];
                            }
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
                            for c in 0..channels {
                                row[dst_idx + c] = src[src_idx + c];
                            }
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
            };

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![out_h, out_w]
            } else {
                vec![out_h, out_w, channels]
            };
            Tensor::from_f32_shape(&out_data, out_shape).unwrap()
        }
        TensorDType::U8 => {
            let bytes = tensor.to_contiguous_bytes();
            let mut f32_src = vec![0.0f32; bytes.len()];
            f32_src
                .par_iter_mut()
                .zip(bytes.par_iter())
                .for_each(|(dst, &b)| *dst = b as f32 / 255.0);

            let f32_tensor = Tensor::from_f32_shape(&f32_src, tensor.shape.to_vec()).unwrap();
            let rotated_f32 = apply_rotate(
                &f32_tensor,
                layout,
                angle_deg,
                expand_canvas,
                fill_value / 255.0,
            );

            let out_bytes = rotated_f32.to_contiguous_bytes();
            let out_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_bytes.len() / 4)
            };
            let mut out_u8 = vec![0u8; out_slice.len()];
            out_u8
                .par_iter_mut()
                .zip(out_slice.par_iter())
                .for_each(|(dst, &f)| {
                    *dst = (f * 255.0).clamp(0.0, 255.0).round() as u8;
                });

            Tensor::from_rvec_u8(
                core_types::RVec::from(out_u8),
                rotated_f32.shape.to_vec(),
                TensorDType::U8,
            )
            .unwrap()
        }
        _ => tensor.clone(),
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
