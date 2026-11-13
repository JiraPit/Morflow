use core_types::numeric::{Element, FloatElement, IntElement};
use core_types::shapecheck::PreparedArgs;
use core_types::{match_dtype, DataType, Payload, Shape, ShapeResult, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg};
    let rank = input.rank();
    let result = contract::finish((|| {
        if input.rank() < 2 {
            return Err("pool2d requires rank at least 2".into());
        }
        let kernel = arg::<usize>(&args, &["kernel_size", "kernel"], Some(0), Some(2))?
            .unwrap()
            .max(1);
        let stride = arg::<usize>(&args, &["stride"], Some(1), Some(2))?
            .unwrap()
            .max(1);
        let mut out = input.dims().to_vec();
        let rank = out.len();
        for d in &mut out[rank - 2..] {
            if d.is_unknown() {
                continue;
            }
            if *d < kernel {
                return Err("Pooling kernel exceeds input spatial dimension".into());
            }
            *d = ((*d).known().unwrap() - kernel)
                .checked_div(stride)
                .unwrap()
                .checked_add(1)
                .unwrap()
                .into();
        }
        contract::shape(out)
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

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let inner_payload = payload.into_unwrapped();
    let kernel_size = prepared
        .unsigned("kernel")
        .expect("shapecheck prepared kernel") as usize;
    let stride = prepared
        .unsigned("stride")
        .expect("shapecheck prepared stride") as usize;

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            let geometry = Geometry::new(
                &tensor.shape,
                kernel_size,
                stride,
                prepared
                    .output_dims()
                    .expect("shapecheck prepared dimensions"),
            );
            let r = tensor.rank();
            let in_plane = tensor.shape[r - 2] * tensor.shape[r - 1];
            // Max pooling keeps the dtype; NaN propagates like NumPy.
            let result = match_dtype!(tensor.dtype,
                int I => {
                    let vals = tensor.to_vec_of::<I>().unwrap_or_default();
                    let out = geometry.pool(&vals, in_plane, I::from_i128(I::MIN_I128), |a, b| a.max(b), |a| a);
                    Tensor::from_vec_of(out, geometry.out_shape.clone())
                },
                float F => {
                    let vals = tensor.to_vec_of::<F>().unwrap_or_default();
                    let out = geometry.pool(&vals, in_plane, F::NEG_INFINITY, |a, b| a.nan_max(b), |a| a);
                    Tensor::from_vec_of(out, geometry.out_shape.clone())
                },
            );
            match result {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'max_pool2d\' requires a tensor or scalar value",
        )),
    }
}

/// Window layout of a 2D pooling over the last two axes.
struct Geometry {
    k: usize,
    s: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    batch_size: usize,
    out_shape: Vec<usize>,
}

impl Geometry {
    fn new(shape: &[usize], k: usize, s: usize, out_shape: Vec<usize>) -> Self {
        let r = shape.len();
        Self {
            k,
            s,
            w: shape[r - 1],
            out_h: out_shape[r - 2],
            out_w: out_shape[r - 1],
            batch_size: shape[0..r - 2].iter().product(),
            out_shape,
        }
    }

    /// Folds every window with `combine`, starting from `init`.
    fn pool<T: Copy + Send + Sync + Default>(
        &self,
        vals: &[T],
        in_plane_size: usize,
        init: T,
        combine: impl Fn(T, T) -> T + Sync + Send,
        finish: impl Fn(T) -> T + Sync + Send,
    ) -> Vec<T> {
        let (k, s, w, out_h, out_w) = (self.k, self.s, self.w, self.out_h, self.out_w);
        let out_plane_size = out_h * out_w;
        let mut out_vals = vec![T::default(); self.batch_size * out_plane_size];
        if out_vals.is_empty() {
            return out_vals;
        }
        out_vals
            .par_chunks_mut(out_plane_size)
            .enumerate()
            .for_each(|(b, out_plane)| {
                let in_plane_offset = b * in_plane_size;
                for oh in 0..out_h {
                    let ih_start = oh * s;
                    for ow in 0..out_w {
                        let iw_start = ow * s;
                        let mut acc = init;
                        for kh in 0..k {
                            let ih = ih_start + kh;
                            for kw in 0..k {
                                let iw = iw_start + kw;
                                let idx = in_plane_offset + ih * w + iw;
                                acc = combine(acc, vals[idx]);
                            }
                        }
                        out_plane[oh * out_w + ow] = finish(acc);
                    }
                }
            });
        out_vals
    }
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    let result = core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    );
    core_types::shapecheck::pool2d_plan(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_max_pool2d_action() {
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let tensor = Tensor::from_f32_shape(&data, vec![4, 4]).unwrap();

        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[6.0, 8.0, 14.0, 16.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
