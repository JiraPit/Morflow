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

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let mut strength = 1.0f32;
    let mut sigma = 1.0f32;
    let mut radius_opt: Option<usize> = None;

    if let Some(args) = args_opt {
        if let Some(st_str) = args
            .get_named("strength")
            .or_else(|| args.get_named("amount"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(st) = st_str.parse::<f32>() {
                strength = st;
            }
        }
        if let Some(s_str) = args.get_named("sigma") {
            if let Ok(s) = s_str.parse::<f32>() {
                sigma = s.max(0.01);
            }
        }
        if let Some(r_str) = args.get_named("radius") {
            if let Ok(r) = r_str.parse::<usize>() {
                radius_opt = Some(r);
            }
        }
    }

    let radius = radius_opt.unwrap_or_else(|| (3.0 * sigma).ceil().max(1.0) as usize);

    match inner_payload {
        Payload::Tensor(tensor) => {
            let layout = if tensor.shape.len() == 3 && tensor.shape[2] <= 4 {
                ImageLayout::Hwc
            } else if tensor.shape.len() == 3 && tensor.shape[0] <= 4 {
                ImageLayout::Chw
            } else {
                ImageLayout::Hwc
            };
            let res = apply_sharpen(&tensor, layout, strength, sigma, radius);
            Payload::Tensor(res)
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'sharpen\' requires Payload::Tensor",
        )),
    }
}

fn apply_sharpen(
    tensor: &Tensor,
    layout: ImageLayout,
    strength: f32,
    sigma: f32,
    radius: usize,
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

            let blurred = separable_gaussian_f32(src, width, height, channels, sigma, radius);
            let mut sharp = vec![0.0f32; src.len()];
            sharp
                .par_iter_mut()
                .zip(src.par_iter())
                .zip(blurred.par_iter())
                .for_each(|((dst, &orig), &blur)| {
                    *dst = orig + strength * (orig - blur);
                });

            let out_shape = if channels == 1 && shape.len() == 2 {
                vec![height, width]
            } else if layout == ImageLayout::Hwc {
                vec![height, width, channels]
            } else {
                vec![channels, height, width]
            };
            Tensor::from_f32_vec(sharp, out_shape).unwrap()
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

            let blurred = separable_gaussian_f32(&src_f32, width, height, channels, sigma, radius);
            let mut sharp_f32 = vec![0.0f32; src_f32.len()];
            sharp_f32
                .par_iter_mut()
                .zip(src_f32.par_iter())
                .zip(blurred.par_iter())
                .for_each(|((dst, &orig), &blur)| {
                    *dst = orig + strength * (orig - blur);
                });

            let mut out_u8 = vec![0u8; sharp_f32.len()];
            out_u8
                .par_iter_mut()
                .zip(sharp_f32.par_iter())
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

fn generate_gaussian_kernel_1d(sigma: f32, radius: usize) -> Vec<f32> {
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0f32; size];
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;

    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        let v = (-x * x / two_sigma_sq).exp();
        *k = v;
        sum += v;
    }

    if sum > 0.0 {
        for k in &mut kernel {
            *k /= sum;
        }
    }

    kernel
}

fn separable_gaussian_f32(
    src: &[f32],
    w: usize,
    h: usize,
    c: usize,
    sigma: f32,
    radius: usize,
) -> Vec<f32> {
    let kernel = generate_gaussian_kernel_1d(sigma, radius);
    let mut temp = vec![0.0f32; h * w * c];
    let mut out = vec![0.0f32; h * w * c];

    // Horizontal pass
    temp.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            let src_row = &src[y * w * c..(y + 1) * w * c];
            for x in 0..w {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for (k_idx, &weight) in kernel.iter().enumerate() {
                        let sample_x = (x as isize + k_idx as isize - radius as isize)
                            .clamp(0, w as isize - 1)
                            as usize;
                        sum += weight * src_row[sample_x * c + ch];
                    }
                    row[x * c + ch] = sum;
                }
            }
        });

    // Vertical pass
    out.par_chunks_exact_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for (k_idx, &weight) in kernel.iter().enumerate() {
                        let sample_y = (y as isize + k_idx as isize - radius as isize)
                            .clamp(0, h as isize - 1)
                            as usize;
                        sum += weight * temp[(sample_y * w + x) * c + ch];
                    }
                    row[x * c + ch] = sum;
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
    fn test_sharpen_edge_enhancement() {
        let f32_data = vec![0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5];
        let tensor = Tensor::from_f32_shape(&f32_data, vec![3, 3]).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("strength"), RString::from("1.5")));

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
            // Sharpen increases local contrast: center is higher than original 1.0
            assert!(slice[4] > 1.0);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
