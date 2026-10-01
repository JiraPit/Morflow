#![allow(clippy::manual_memcpy)]

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
    let mut flip_h = false;
    let mut flip_v = false;

    if let Some(args) = args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            match ax_str.to_lowercase().trim() {
                "horizontal" | "h" | "x" | "1" => flip_h = true,
                "vertical" | "v" | "y" | "0" => flip_v = true,
                "both" | "hv" | "xy" => {
                    flip_h = true;
                    flip_v = true;
                }
                _ => {}
            }
        }
    }

    if !flip_h && !flip_v {
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
            let res = apply_flip(&tensor, layout, flip_h, flip_v);
            Payload::Tensor(res)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'flip\' requires Payload::Tensor",
        )),
    }
}

fn apply_flip(tensor: &Tensor, layout: ImageLayout, flip_h: bool, flip_v: bool) -> Tensor {
    let shape = tensor.shape.as_slice();
    let (in_h, in_w, channels) = match (shape.len(), layout) {
        (2, _) => (shape[0], shape[1], 1),
        (3, ImageLayout::Hwc) => (shape[0], shape[1], shape[2]),
        (3, ImageLayout::Chw) => (shape[1], shape[2], shape[0]),
        _ => return tensor.clone(),
    };

    match tensor.dtype {
        TensorDType::F32 => {
            let Some(src) = tensor.as_f32_slice() else {
                return tensor.clone();
            };
            let mut out = vec![0.0f32; in_h * in_w * channels];

            match layout {
                ImageLayout::Hwc => {
                    out.par_chunks_exact_mut(in_w * channels)
                        .enumerate()
                        .for_each(|(y, row)| {
                            let sy = if flip_v { in_h - 1 - y } else { y };
                            for x in 0..in_w {
                                let sx = if flip_h { in_w - 1 - x } else { x };
                                let src_idx = (sy * in_w + sx) * channels;
                                let dst_idx = x * channels;
                                for c in 0..channels {
                                    row[dst_idx + c] = src[src_idx + c];
                                }
                            }
                        });
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![in_h, in_w]
                    } else {
                        vec![in_h, in_w, channels]
                    };
                    Tensor::from_f32_vec(out, out_shape).unwrap()
                }
                ImageLayout::Chw => {
                    let plane_size = in_h * in_w;
                    out.par_chunks_exact_mut(plane_size)
                        .enumerate()
                        .for_each(|(c, plane)| {
                            let src_plane = &src[c * plane_size..(c + 1) * plane_size];
                            for y in 0..in_h {
                                let sy = if flip_v { in_h - 1 - y } else { y };
                                for x in 0..in_w {
                                    let sx = if flip_h { in_w - 1 - x } else { x };
                                    plane[y * in_w + x] = src_plane[sy * in_w + sx];
                                }
                            }
                        });
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![in_h, in_w]
                    } else {
                        vec![channels, in_h, in_w]
                    };
                    Tensor::from_f32_vec(out, out_shape).unwrap()
                }
            }
        }
        TensorDType::U8 => {
            let Some(bytes) = tensor.as_u8_slice() else {
                return tensor.clone();
            };
            let mut out = vec![0u8; in_h * in_w * channels];

            match layout {
                ImageLayout::Hwc => {
                    out.par_chunks_exact_mut(in_w * channels)
                        .enumerate()
                        .for_each(|(y, row)| {
                            let sy = if flip_v { in_h - 1 - y } else { y };
                            for x in 0..in_w {
                                let sx = if flip_h { in_w - 1 - x } else { x };
                                let src_idx = (sy * in_w + sx) * channels;
                                let dst_idx = x * channels;
                                for c in 0..channels {
                                    row[dst_idx + c] = bytes[src_idx + c];
                                }
                            }
                        });
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![in_h, in_w]
                    } else {
                        vec![in_h, in_w, channels]
                    };
                    Tensor::from_rvec_u8(core_types::RVec::from(out), out_shape, TensorDType::U8)
                        .unwrap()
                }
                ImageLayout::Chw => {
                    let plane_size = in_h * in_w;
                    out.par_chunks_exact_mut(plane_size)
                        .enumerate()
                        .for_each(|(c, plane)| {
                            let src_plane = &bytes[c * plane_size..(c + 1) * plane_size];
                            for y in 0..in_h {
                                let sy = if flip_v { in_h - 1 - y } else { y };
                                for x in 0..in_w {
                                    let sx = if flip_h { in_w - 1 - x } else { x };
                                    plane[y * in_w + x] = src_plane[sy * in_w + sx];
                                }
                            }
                        });
                    let out_shape = if channels == 1 && shape.len() == 2 {
                        vec![in_h, in_w]
                    } else {
                        vec![channels, in_h, in_w]
                    };
                    Tensor::from_rvec_u8(core_types::RVec::from(out), out_shape, TensorDType::U8)
                        .unwrap()
                }
            }
        }
        _ => tensor.clone(),
    }
}

#[no_mangle]
pub extern "C" fn get_action_abi_version() -> u32 {
    core_types::shapecheck::ACTION_ABI_VERSION
}
#[no_mangle]
pub extern "C" fn get_action_abi_layout() -> *const core_types::abi_stable::type_layout::TypeLayout
{
    <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT
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
    fn test_flip_horizontal() {
        let f32_data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![2, 2]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("axis"), RString::from("horizontal")));

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
            assert_eq!(slice, &[2.0, 1.0, 4.0, 3.0]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }

    #[test]
    fn test_flip_vertical_positional() {
        let f32_data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![2, 2]).unwrap();

        let mut positional = core_types::RVec::new();
        positional.push(RString::from("vertical"));

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Tensor(tensor)),
            args: ActionArgs {
                positional,
                named: core_types::RVec::new(),
            },
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            let slice: &[f32] = out_t.as_f32_slice().unwrap();
            assert_eq!(slice, &[3.0, 4.0, 1.0, 2.0]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
