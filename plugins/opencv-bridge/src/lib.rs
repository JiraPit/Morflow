//! Prebuilt adapter to shared system OpenCV. No OpenCV code is vendored.
use morflow_opencv::Options;
use opencv::{
    core::{self, Mat, Point, Scalar, Size},
    imgproc,
    prelude::*,
};
use std::ffi::{c_char, c_void};

fn invalid(message: &str) -> opencv::Error {
    opencv::Error::new(core::StsBadArg, message)
}

/// Process contiguous HWC buffers owned by the action.
///
/// # Safety
/// Input and output must point to separate buffers sized for the given positive
/// dimensions, channels and depth. All pointers remain valid for this call.
#[no_mangle]
pub unsafe extern "C" fn morflow_opencv_process(
    input: *const c_void,
    output: *mut c_void,
    height: i32,
    width: i32,
    channels: i32,
    depth: i32,
    out_height: i32,
    out_width: i32,
    options: *const Options,
    error: *mut c_char,
    error_length: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if input.is_null()
            || output.is_null()
            || options.is_null()
            || height <= 0
            || width <= 0
            || out_height <= 0
            || out_width <= 0
            || !(1..=512).contains(&channels)
            || ![core::CV_8U, core::CV_32F].contains(&depth)
        {
            return Err(invalid("Invalid image buffer descriptor"));
        }
        let o = &*options;
        if o.operation != 0 && o.operation != 5 && (height != out_height || width != out_width) {
            return Err(invalid(
                "Shape-preserving operation has mismatched dimensions",
            ));
        }
        if o.radius < 0
            || o.radius > (i32::MAX - 1) / 2
            || o.iterations < 1
            || !o.sigma.is_finite()
            || o.sigma <= 0.
            || !o.strength.is_finite()
            || !o.angle.is_finite()
            || !o.fill.is_finite()
        {
            return Err(invalid("Invalid image processing arguments"));
        }
        let src = Mat::new_rows_cols_with_data_unsafe_def(
            height,
            width,
            core::CV_MAKETYPE(depth, channels),
            input.cast_mut(),
        )?;
        let mut dst =
            Mat::new_rows_cols_with_data_unsafe_def(out_height, out_width, src.typ(), output)?;
        let size = Size::new(out_width, out_height);
        let anchor = Point::new(-1, -1);
        if o.operation == 0 {
            imgproc::resize(&src, &mut dst, size, 0., 0., o.mode)?;
        } else if o.operation == 5 {
            let rotation = if (o.angle - 90.).abs() < 1e-3 {
                Some(core::ROTATE_90_CLOCKWISE)
            } else if (o.angle - 180.).abs() < 1e-3 {
                Some(core::ROTATE_180)
            } else if (o.angle - 270.).abs() < 1e-3 {
                Some(core::ROTATE_90_COUNTERCLOCKWISE)
            } else {
                None
            };
            if let Some(rotation) = rotation {
                core::rotate(&src, &mut dst, rotation)?;
            } else if o.angle.abs() < 1e-3 {
                src.copy_to(&mut dst)?;
            } else {
                let (s, c) = o.angle.to_radians().sin_cos();
                let matrix = Mat::from_slice_2d(&[
                    [
                        c,
                        s,
                        width as f64 / 2. - c * out_width as f64 / 2. - s * out_height as f64 / 2.,
                    ],
                    [
                        -s,
                        c,
                        height as f64 / 2. + s * out_width as f64 / 2. - c * out_height as f64 / 2.,
                    ],
                ])?;
                imgproc::warp_affine(
                    &src,
                    &mut dst,
                    &matrix,
                    size,
                    imgproc::INTER_LINEAR | imgproc::WARP_INVERSE_MAP,
                    core::BORDER_CONSTANT,
                    Scalar::all(o.fill),
                )?;
            }
        } else if o.operation == 2 {
            if o.radius == 0 {
                src.copy_to(&mut dst)?;
            } else {
                let side = 2 * o.radius + 1;
                let mut element =
                    Mat::new_rows_cols_with_default(side, side, core::CV_8U, Scalar::all(0.))?;
                for y in -o.radius..=o.radius {
                    for x in -o.radius..=o.radius {
                        if o.shape == 0
                            || (o.shape == 1 && (x == 0 || y == 0))
                            || (o.shape == 2
                                && i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y)
                                    <= i64::from(o.radius) * i64::from(o.radius))
                        {
                            *element.at_2d_mut::<u8>(y + o.radius, x + o.radius)? = 1;
                        }
                    }
                }
                imgproc::morphology_ex(
                    &src,
                    &mut dst,
                    o.mode,
                    &element,
                    anchor,
                    if o.mode == imgproc::MORPH_GRADIENT {
                        1
                    } else {
                        o.iterations
                    },
                    core::BORDER_REPLICATE,
                    imgproc::morphology_default_border_value()?,
                )?;
            }
        } else {
            let mut converted = Mat::default();
            let values = if depth == core::CV_8U {
                src.convert_to(&mut converted, core::CV_32F, 1. / 255., 0.)?;
                &converted
            } else {
                &src
            };
            // Separate Mat headers borrow the same caller-owned destination.
            let mut result = if depth == core::CV_32F {
                Mat::new_rows_cols_with_data_unsafe_def(out_height, out_width, src.typ(), output)?
            } else {
                Mat::default()
            };
            let side = Size::new(2 * o.radius + 1, 2 * o.radius + 1);
            if o.operation == 1 || o.operation == 4 {
                let mut blurred = if o.operation == 1 && depth == core::CV_32F {
                    Mat::new_rows_cols_with_data_unsafe_def(
                        out_height,
                        out_width,
                        src.typ(),
                        output,
                    )?
                } else {
                    Mat::default()
                };
                if o.operation == 1 && o.mode == 1 {
                    imgproc::blur(values, &mut blurred, side, anchor, core::BORDER_REPLICATE)?;
                } else {
                    #[cfg(opencv_gaussian_hint)]
                    imgproc::gaussian_blur(
                        values,
                        &mut blurred,
                        side,
                        o.sigma,
                        o.sigma,
                        core::BORDER_REPLICATE,
                        core::AlgorithmHint::ALGO_HINT_DEFAULT,
                    )?;
                    #[cfg(not(opencv_gaussian_hint))]
                    imgproc::gaussian_blur(
                        values,
                        &mut blurred,
                        side,
                        o.sigma,
                        o.sigma,
                        core::BORDER_REPLICATE,
                    )?;
                }
                if o.operation == 4 {
                    core::add_weighted(
                        values,
                        1. + o.strength,
                        &blurred,
                        -o.strength,
                        0.,
                        &mut result,
                        -1,
                    )?;
                } else {
                    blurred.copy_to(&mut result)?;
                }
            } else if o.operation == 3 {
                if o.mode == 3 || o.mode == 1 || o.mode == 2 {
                    let mut derivative = Mat::default();
                    if o.mode == 3 {
                        imgproc::laplacian(
                            values,
                            &mut derivative,
                            core::CV_32F,
                            1,
                            o.strength,
                            0.,
                            core::BORDER_REPLICATE,
                        )?;
                    } else {
                        imgproc::sobel(
                            values,
                            &mut derivative,
                            core::CV_32F,
                            i32::from(o.mode == 1),
                            i32::from(o.mode == 2),
                            3,
                            o.strength,
                            0.,
                            core::BORDER_REPLICATE,
                        )?;
                    }
                    let zero = Mat::zeros(derivative.rows(), derivative.cols(), derivative.typ())?
                        .to_mat()?;
                    core::absdiff(&derivative, &zero, &mut result)?;
                } else {
                    let mut gx = Mat::default();
                    let mut gy = Mat::default();
                    if o.mode == 4 {
                        let smooth = Mat::from_slice(&[1f32, 1., 1.])?;
                        let derivative = Mat::from_slice(&[-1f32, 0., 1.])?;
                        imgproc::sep_filter_2d(
                            values,
                            &mut gx,
                            core::CV_32F,
                            &derivative,
                            &smooth,
                            anchor,
                            0.,
                            core::BORDER_REPLICATE,
                        )?;
                        imgproc::sep_filter_2d(
                            values,
                            &mut gy,
                            core::CV_32F,
                            &smooth,
                            &derivative,
                            anchor,
                            0.,
                            core::BORDER_REPLICATE,
                        )?;
                    } else {
                        imgproc::sobel(
                            values,
                            &mut gx,
                            core::CV_32F,
                            1,
                            0,
                            3,
                            1.,
                            0.,
                            core::BORDER_REPLICATE,
                        )?;
                        imgproc::sobel(
                            values,
                            &mut gy,
                            core::CV_32F,
                            0,
                            1,
                            3,
                            1.,
                            0.,
                            core::BORDER_REPLICATE,
                        )?;
                    }
                    core::magnitude(&gx, &gy, &mut result)?;
                    if o.strength != 1. {
                        let source = Mat::new_rows_cols_with_data_unsafe_def(
                            out_height,
                            out_width,
                            result.typ(),
                            result.data_mut().cast(),
                        )?;
                        source.convert_to(&mut result, core::CV_32F, o.strength, 0.)?;
                    }
                }
            } else {
                return Err(invalid("Unknown image operation"));
            }
            if depth == core::CV_8U {
                result.convert_to(&mut dst, core::CV_8U, 255., 0.)?;
            }
        }
        Ok::<(), opencv::Error>(())
    });
    let message = match result {
        Ok(Ok(())) => return 0,
        Ok(Err(e)) => e.to_string(),
        Err(_) => "OpenCV plugin processing panicked".into(),
    };
    if !error.is_null() && error_length > 0 {
        let bytes = message.as_bytes();
        let count = bytes.len().min(error_length - 1);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), error.cast(), count);
        *error.add(count) = 0;
    }
    1
}
