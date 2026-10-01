use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, ImageLayout, Payload, Shape, ShapeResult, Tensor, TensorDType};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if !matches!(input.rank(), 2 | 3) {
            return Err("Image operations require rank 2 or 3".into());
        }
        contract::image_nonempty(&input)?;
        Ok(input)
    })())
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
    let payload = match core_types::contract::image_input(payload, true) {
        Ok(payload) => payload,
        Err(error) => return Payload::Error(error),
    };
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));
    let mut brightness = 0.0f32;
    let mut contrast = 1.0f32;
    let mut gamma = 1.0f32;
    let mut saturation = 1.0f32;
    let mut exposure = 0.0f32;

    if let Some(args) = &args_opt {
        if let Some(b_str) = args.get_named("brightness") {
            if let Ok(b) = prepared.args.parse::<f32>(b_str) {
                brightness = b;
            }
        }
        if let Some(c_str) = args.get_named("contrast") {
            if let Ok(c) = prepared.args.parse::<f32>(c_str) {
                contrast = c;
            }
        }
        if let Some(g_str) = args.get_named("gamma") {
            if let Ok(g) = prepared.args.parse::<f32>(g_str) {
                gamma = g.max(0.001);
            }
        }
        if let Some(s_str) = args
            .get_named("saturation")
            .or_else(|| args.get_named("sat"))
        {
            if let Ok(s) = prepared.args.parse::<f32>(s_str) {
                saturation = s;
            }
        }
        if let Some(e_str) = args.get_named("exposure") {
            if let Ok(e) = prepared.args.parse::<f32>(e_str) {
                exposure = e;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(mut tensor) => {
            let layout = if tensor.shape.len() == 3 && tensor.shape[2] <= 4 {
                ImageLayout::Hwc
            } else if tensor.shape.len() == 3 && tensor.shape[0] <= 4 {
                ImageLayout::Chw
            } else {
                ImageLayout::Hwc
            };
            adjust_color_tensor(
                &mut tensor,
                layout,
                brightness,
                contrast,
                gamma,
                saturation,
                exposure,
            );
            Payload::Tensor(tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'color_adjust\' requires Payload::Tensor",
        )),
    }
}

fn adjust_color_tensor(
    tensor: &mut Tensor,
    layout: ImageLayout,
    brightness: f32,
    contrast: f32,
    gamma: f32,
    saturation: f32,
    exposure: f32,
) {
    let shape = tensor.shape.as_slice();
    let (height, width, channels) = match (shape.len(), layout) {
        (2, _) => (shape[0], shape[1], 1),
        (3, ImageLayout::Hwc) => (shape[0], shape[1], shape[2]),
        (3, ImageLayout::Chw) => (shape[1], shape[2], shape[0]),
        _ => return,
    };

    let exposure_mult = 2.0f32.powf(exposure);
    let inv_gamma = 1.0f32 / gamma;

    match tensor.dtype {
        TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();

            if layout == ImageLayout::Hwc {
                slice.par_chunks_exact_mut(channels).for_each(|pixel| {
                    if channels == 3 || channels == 4 {
                        let mut r = (pixel[0] * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        let mut g = (pixel[1] * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        let mut b = (pixel[2] * exposure_mult - 0.5) * contrast + 0.5 + brightness;

                        if gamma != 1.0 {
                            r = r.max(0.0).powf(inv_gamma);
                            g = g.max(0.0).powf(inv_gamma);
                            b = b.max(0.0).powf(inv_gamma);
                        }

                        if saturation != 1.0 {
                            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                            r = luma + (r - luma) * saturation;
                            g = luma + (g - luma) * saturation;
                            b = luma + (b - luma) * saturation;
                        }

                        pixel[0] = r;
                        pixel[1] = g;
                        pixel[2] = b;
                    } else if channels == 1 {
                        let mut val =
                            (pixel[0] * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        if gamma != 1.0 {
                            val = val.max(0.0).powf(inv_gamma);
                        }
                        pixel[0] = val;
                    }
                });
            } else {
                // CHW layout
                let plane_size = height * width;
                if channels >= 3 {
                    let (r_plane, rest) = slice.split_at_mut(plane_size);
                    let (g_plane, b_plane) = rest.split_at_mut(plane_size);

                    r_plane
                        .par_iter_mut()
                        .zip(g_plane.par_iter_mut())
                        .zip(b_plane.par_iter_mut())
                        .for_each(|((r, g), b)| {
                            let r_in = *r;
                            let g_in = *g;
                            let b_in = *b;

                            let mut r_val =
                                (r_in * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                            let mut g_val =
                                (g_in * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                            let mut b_val =
                                (b_in * exposure_mult - 0.5) * contrast + 0.5 + brightness;

                            if gamma != 1.0 {
                                r_val = r_val.max(0.0).powf(inv_gamma);
                                g_val = g_val.max(0.0).powf(inv_gamma);
                                b_val = b_val.max(0.0).powf(inv_gamma);
                            }

                            if saturation != 1.0 {
                                let luma = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val;
                                r_val = luma + (r_val - luma) * saturation;
                                g_val = luma + (g_val - luma) * saturation;
                                b_val = luma + (b_val - luma) * saturation;
                            }

                            *r = r_val;
                            *g = g_val;
                            *b = b_val;
                        });
                } else {
                    slice.par_iter_mut().for_each(|val| {
                        let mut v = (*val * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        if gamma != 1.0 {
                            v = v.max(0.0).powf(inv_gamma);
                        }
                        *val = v;
                    });
                }
            }
        }
        TensorDType::U8 => {
            let slice = tensor.as_u8_slice_mut();

            if layout == ImageLayout::Hwc {
                slice.par_chunks_exact_mut(channels).for_each(|pixel| {
                    if channels == 3 || channels == 4 {
                        let rf = pixel[0] as f32 / 255.0;
                        let gf = pixel[1] as f32 / 255.0;
                        let bf = pixel[2] as f32 / 255.0;

                        let mut r = (rf * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        let mut g = (gf * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        let mut b = (bf * exposure_mult - 0.5) * contrast + 0.5 + brightness;

                        if gamma != 1.0 {
                            r = r.max(0.0).powf(inv_gamma);
                            g = g.max(0.0).powf(inv_gamma);
                            b = b.max(0.0).powf(inv_gamma);
                        }

                        if saturation != 1.0 {
                            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                            r = luma + (r - luma) * saturation;
                            g = luma + (g - luma) * saturation;
                            b = luma + (b - luma) * saturation;
                        }

                        pixel[0] = (r * 255.0).clamp(0.0, 255.0).round() as u8;
                        pixel[1] = (g * 255.0).clamp(0.0, 255.0).round() as u8;
                        pixel[2] = (b * 255.0).clamp(0.0, 255.0).round() as u8;
                    } else if channels == 1 {
                        let vf = pixel[0] as f32 / 255.0;
                        let mut val = (vf * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                        if gamma != 1.0 {
                            val = val.max(0.0).powf(inv_gamma);
                        }
                        pixel[0] = (val * 255.0).clamp(0.0, 255.0).round() as u8;
                    }
                });
            } else {
                // CHW U8
                slice.par_iter_mut().for_each(|val| {
                    let vf = *val as f32 / 255.0;
                    let mut v = (vf * exposure_mult - 0.5) * contrast + 0.5 + brightness;
                    if gamma != 1.0 {
                        v = v.max(0.0).powf(inv_gamma);
                    }
                    *val = (v * 255.0).clamp(0.0, 255.0).round() as u8;
                });
            }
        }
        _ => {}
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
    use core_types::{ActionArgs, RBox, RString, Tuple2};

    #[test]
    fn test_color_adjust_brightness_contrast() {
        let f32_data = vec![0.5f32, 0.5, 0.5];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![1, 1, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("brightness"), RString::from("0.2")));
        named.push(Tuple2(RString::from("contrast"), RString::from("1.5")));

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
            assert!((slice[0] - 0.7).abs() < 1e-4);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
