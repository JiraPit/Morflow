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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlendMode {
    Alpha,
    Multiply,
    Screen,
    Overlay,
    Add,
    Subtract,
    Difference,
    Darken,
    Lighten,
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let mut mode = BlendMode::Alpha;
    let mut opacity = 1.0f32;
    let mut solid_color: Option<Vec<f32>> = None;

    if let Some(args) = payload.args() {
        if let Some(m_str) = args
            .get_named("mode")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            mode = match m_str.to_lowercase().as_str() {
                "multiply" | "mult" => BlendMode::Multiply,
                "screen" => BlendMode::Screen,
                "overlay" => BlendMode::Overlay,
                "add" | "addition" => BlendMode::Add,
                "subtract" | "sub" => BlendMode::Subtract,
                "difference" | "diff" => BlendMode::Difference,
                "darken" => BlendMode::Darken,
                "lighten" => BlendMode::Lighten,
                _ => BlendMode::Alpha,
            };
        }
        if let Some(op_str) = args
            .get_named("opacity")
            .or_else(|| args.get_named("alpha"))
        {
            if let Ok(op) = op_str.parse::<f32>() {
                opacity = op.clamp(0.0, 1.0);
            }
        }
        if let Some(c_str) = args.get_named("color") {
            // e.g. "1.0,0.5,0.2" or "255,128,64"
            let parsed: Vec<f32> = c_str
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .map(|v| if v > 1.0 { v / 255.0 } else { v })
                .collect();
            if !parsed.is_empty() {
                solid_color = Some(parsed);
            }
        }
    }

    match payload.unwrap_payload() {
        Payload::Image(img) => {
            let res = apply_blend(
                &img.tensor,
                img.layout,
                mode,
                opacity,
                solid_color.as_deref(),
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
            let res = apply_blend(tensor, layout, mode, opacity, solid_color.as_deref());
            Payload::Tensor(res)
        }
        other => other.clone(),
    }
}

fn apply_blend(
    tensor: &Tensor,
    layout: ImageLayout,
    mode: BlendMode,
    opacity: f32,
    solid_color: Option<&[f32]>,
) -> Tensor {
    let shape = tensor.shape.as_slice();
    let channels = match (shape.len(), layout) {
        (2, _) => 1,
        (3, ImageLayout::Hwc) => shape[2],
        (3, ImageLayout::Chw) => shape[0],
        _ => return tensor.clone(),
    };

    match tensor.dtype {
        TensorDType::F32 => {
            let mut bytes = tensor.to_contiguous_bytes();
            let slice: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4)
            };

            let default_color = vec![1.0f32; channels];
            let fg_color = solid_color.unwrap_or(&default_color);

            slice.par_chunks_exact_mut(channels).for_each(|pixel| {
                for c in 0..channels {
                    let a = pixel[c];
                    let b = *fg_color.get(c).unwrap_or(&1.0);

                    let blended = match mode {
                        BlendMode::Alpha => b,
                        BlendMode::Multiply => a * b,
                        BlendMode::Screen => 1.0 - (1.0 - a) * (1.0 - b),
                        BlendMode::Overlay => {
                            if a < 0.5 {
                                2.0 * a * b
                            } else {
                                1.0 - 2.0 * (1.0 - a) * (1.0 - b)
                            }
                        }
                        BlendMode::Add => (a + b).min(1.0),
                        BlendMode::Subtract => (a - b).max(0.0),
                        BlendMode::Difference => (a - b).abs(),
                        BlendMode::Darken => a.min(b),
                        BlendMode::Lighten => a.max(b),
                    };

                    pixel[c] = (1.0 - opacity) * a + opacity * blended;
                }
            });

            Tensor::from_f32_shape(slice, tensor.shape.to_vec()).unwrap()
        }
        TensorDType::U8 => {
            let mut bytes = tensor.to_contiguous_bytes();
            let slice: &mut [u8] = bytes.as_mut_slice();

            let default_color = vec![1.0f32; channels];
            let fg_color = solid_color.unwrap_or(&default_color);

            slice.par_chunks_exact_mut(channels).for_each(|pixel| {
                for c in 0..channels {
                    let a = pixel[c] as f32 / 255.0;
                    let b = *fg_color.get(c).unwrap_or(&1.0);

                    let blended = match mode {
                        BlendMode::Alpha => b,
                        BlendMode::Multiply => a * b,
                        BlendMode::Screen => 1.0 - (1.0 - a) * (1.0 - b),
                        BlendMode::Overlay => {
                            if a < 0.5 {
                                2.0 * a * b
                            } else {
                                1.0 - 2.0 * (1.0 - a) * (1.0 - b)
                            }
                        }
                        BlendMode::Add => (a + b).min(1.0),
                        BlendMode::Subtract => (a - b).max(0.0),
                        BlendMode::Difference => (a - b).abs(),
                        BlendMode::Darken => a.min(b),
                        BlendMode::Lighten => a.max(b),
                    };

                    let res = (1.0 - opacity) * a + opacity * blended;
                    pixel[c] = (res * 255.0).clamp(0.0, 255.0).round() as u8;
                }
            });

            Tensor::from_rvec_u8(bytes, tensor.shape.to_vec(), TensorDType::U8).unwrap()
        }
        _ => tensor.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_blend_multiply() {
        let f32_data = vec![0.5f32, 0.8f32];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![1, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("mode"), RString::from("multiply")));
        named.push(Tuple2(RString::from("color"), RString::from("0.5")));

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
            assert!((slice[0] - 0.25).abs() < 1e-4);
            assert!((slice[1] - 0.4).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
