#![allow(clippy::too_many_arguments, clippy::manual_memcpy)]

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

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg, Error};
    let rank = input.rank();
    let result = contract::finish((|| {
        let (h, w, c, _chw) = contract::image_dims(&input)?;
        let pad = arg::<usize>(&args, &["pad"], Some(0), Some(0))?.unwrap();
        let top = arg::<usize>(&args, &["top", "pad_top"], None, Some(pad))?.unwrap();
        let bottom = arg::<usize>(&args, &["bottom", "pad_bottom"], None, Some(pad))?.unwrap();
        let left = arg::<usize>(&args, &["left", "pad_left"], None, Some(pad))?.unwrap();
        let right = arg::<usize>(&args, &["right", "pad_right"], None, Some(pad))?.unwrap();
        if top == 0 && bottom == 0 && left == 0 && right == 0 {
            return Ok(input);
        }
        let h = h
            .checked_add(top)
            .and_then(|n| n.checked_add(bottom))
            .ok_or(Error::from("Padded height overflows"))?;
        let w = w
            .checked_add(left)
            .and_then(|n| n.checked_add(right))
            .ok_or(Error::from("Padded width overflows"))?;
        // Padding materializes an HWC buffer even for CHW input.
        contract::image_shape(h, w, c, input.rank(), false)
    })());
    match result {
        ShapeResult::Unknown => ShapeResult::Ok(Shape::unknown(rank)),
        other => other,
    }
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PadMode {
    Constant,
    Edge,
    Reflect,
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
    let mut pad_top = 0usize;
    let mut pad_bottom = 0usize;
    let mut pad_left = 0usize;
    let mut pad_right = 0usize;
    let mut pad_mode = PadMode::Constant;
    let mut fill_value = 0.0f32;

    if let Some(args) = args_opt {
        if let Some(p_str) = args
            .get_named("pad")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(p) = prepared.args.parse::<usize>(p_str) {
                pad_top = p;
                pad_bottom = p;
                pad_left = p;
                pad_right = p;
            }
        }
        if let Some(pt_str) = args.get_named("top").or_else(|| args.get_named("pad_top")) {
            if let Ok(p) = prepared.args.parse::<usize>(pt_str) {
                pad_top = p;
            }
        }
        if let Some(pb_str) = args
            .get_named("bottom")
            .or_else(|| args.get_named("pad_bottom"))
        {
            if let Ok(p) = prepared.args.parse::<usize>(pb_str) {
                pad_bottom = p;
            }
        }
        if let Some(pl_str) = args
            .get_named("left")
            .or_else(|| args.get_named("pad_left"))
        {
            if let Ok(p) = prepared.args.parse::<usize>(pl_str) {
                pad_left = p;
            }
        }
        if let Some(pr_str) = args
            .get_named("right")
            .or_else(|| args.get_named("pad_right"))
        {
            if let Ok(p) = prepared.args.parse::<usize>(pr_str) {
                pad_right = p;
            }
        }

        if let Some(pm_str) = args
            .get_named("mode")
            .or_else(|| args.get_named("pad_mode"))
        {
            pad_mode = match pm_str.to_lowercase().as_str() {
                "edge" | "replicate" | "clamp" => PadMode::Edge,
                "reflect" | "mirror" => PadMode::Reflect,
                _ => PadMode::Constant,
            };
        }
        if let Some(fv_str) = args
            .get_named("fill")
            .or_else(|| args.get_named("fill_value"))
        {
            if let Ok(f) = prepared.args.parse::<f32>(fv_str) {
                fill_value = f;
            }
        }
    }

    if pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0 {
        return inner_payload;
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
            let res_tensor = apply_pad(
                &tensor, layout, pad_top, pad_bottom, pad_left, pad_right, pad_mode, fill_value,
            );
            Payload::Tensor(res_tensor)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'pad\' requires Payload::Tensor",
        )),
    }
}

fn apply_pad(
    tensor: &Tensor,
    layout: ImageLayout,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    pad_mode: PadMode,
    fill_value: f32,
) -> Tensor {
    let shape = tensor.shape.as_slice();
    let (in_h, in_w, channels) = match (shape.len(), layout) {
        (2, _) => (shape[0], shape[1], 1),
        (3, ImageLayout::Hwc) => (shape[0], shape[1], shape[2]),
        (3, ImageLayout::Chw) => (shape[1], shape[2], shape[0]),
        _ => return tensor.clone(),
    };

    if pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0 {
        return tensor.clone();
    }

    let out_h = in_h + pad_top + pad_bottom;
    let out_w = in_w + pad_left + pad_right;

    match tensor.dtype {
        TensorDType::F32 => {
            let Some(src) = tensor.as_f32_slice() else {
                return tensor.clone();
            };
            let mut out = vec![fill_value; out_h * out_w * channels];

            out.par_chunks_exact_mut(out_w * channels)
                .enumerate()
                .for_each(|(out_y, row)| {
                    let src_y = if out_y < pad_top {
                        match pad_mode {
                            PadMode::Constant => None,
                            PadMode::Edge => Some(0),
                            PadMode::Reflect => Some((pad_top - out_y).min(in_h - 1)),
                        }
                    } else if out_y >= pad_top + in_h {
                        match pad_mode {
                            PadMode::Constant => None,
                            PadMode::Edge => Some(in_h - 1),
                            PadMode::Reflect => {
                                let diff = out_y - (pad_top + in_h) + 1;
                                Some((in_h - 1).saturating_sub(diff))
                            }
                        }
                    } else {
                        Some(out_y - pad_top)
                    };

                    for out_x in 0..out_w {
                        let src_x = if out_x < pad_left {
                            match pad_mode {
                                PadMode::Constant => None,
                                PadMode::Edge => Some(0),
                                PadMode::Reflect => Some((pad_left - out_x).min(in_w - 1)),
                            }
                        } else if out_x >= pad_left + in_w {
                            match pad_mode {
                                PadMode::Constant => None,
                                PadMode::Edge => Some(in_w - 1),
                                PadMode::Reflect => {
                                    let diff = out_x - (pad_left + in_w) + 1;
                                    Some((in_w - 1).saturating_sub(diff))
                                }
                            }
                        } else {
                            Some(out_x - pad_left)
                        };

                        let dst_idx = out_x * channels;
                        if let (Some(sy), Some(sx)) = (src_y, src_x) {
                            let src_idx = (sy * in_w + sx) * channels;
                            for c in 0..channels {
                                row[dst_idx + c] = src[src_idx + c];
                            }
                        } else {
                            for c in 0..channels {
                                row[dst_idx + c] = fill_value;
                            }
                        }
                    }
                });

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![out_h, out_w]
            } else {
                vec![out_h, out_w, channels]
            };
            Tensor::from_f32_vec(out, out_shape).unwrap()
        }
        TensorDType::U8 => {
            let Some(bytes) = tensor.as_u8_slice() else {
                return tensor.clone();
            };
            let fill_u8 = fill_value.clamp(0.0, 255.0).round() as u8;
            let mut out = vec![fill_u8; out_h * out_w * channels];

            out.par_chunks_exact_mut(out_w * channels)
                .enumerate()
                .for_each(|(out_y, row)| {
                    let src_y = if out_y < pad_top {
                        match pad_mode {
                            PadMode::Constant => None,
                            PadMode::Edge => Some(0),
                            PadMode::Reflect => Some((pad_top - out_y).min(in_h - 1)),
                        }
                    } else if out_y >= pad_top + in_h {
                        match pad_mode {
                            PadMode::Constant => None,
                            PadMode::Edge => Some(in_h - 1),
                            PadMode::Reflect => {
                                let diff = out_y - (pad_top + in_h) + 1;
                                Some((in_h - 1).saturating_sub(diff))
                            }
                        }
                    } else {
                        Some(out_y - pad_top)
                    };

                    for out_x in 0..out_w {
                        let src_x = if out_x < pad_left {
                            match pad_mode {
                                PadMode::Constant => None,
                                PadMode::Edge => Some(0),
                                PadMode::Reflect => Some((pad_left - out_x).min(in_w - 1)),
                            }
                        } else if out_x >= pad_left + in_w {
                            match pad_mode {
                                PadMode::Constant => None,
                                PadMode::Edge => Some(in_w - 1),
                                PadMode::Reflect => {
                                    let diff = out_x - (pad_left + in_w) + 1;
                                    Some((in_w - 1).saturating_sub(diff))
                                }
                            }
                        } else {
                            Some(out_x - pad_left)
                        };

                        let dst_idx = out_x * channels;
                        if let (Some(sy), Some(sx)) = (src_y, src_x) {
                            let src_idx = (sy * in_w + sx) * channels;
                            for c in 0..channels {
                                row[dst_idx + c] = bytes[src_idx + c];
                            }
                        } else {
                            for c in 0..channels {
                                row[dst_idx + c] = fill_u8;
                            }
                        }
                    }
                });

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![out_h, out_w]
            } else {
                vec![out_h, out_w, channels]
            };
            Tensor::from_rvec_u8(core_types::RVec::from(out), out_shape, TensorDType::U8).unwrap()
        }
        _ => tensor.clone(),
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
    fn test_pad_constant() {
        let f32_data = vec![5.0f32];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![1, 1]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("pad"), RString::from("1")));
        named.push(Tuple2(RString::from("fill"), RString::from("0.0")));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional: core_types::RVec::new(),
                named,
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[3, 3]);
            let slice: &[f32] = out_t.as_f32_slice().unwrap();
            assert_eq!(slice[4], 5.0); // Center is 5.0
            assert_eq!(slice[0], 0.0); // Border is 0.0
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
