use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
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
            match avg_pool2d_tensor(
                &tensor,
                kernel_size,
                stride,
                prepared
                    .output_dims()
                    .expect("shapecheck prepared dimensions"),
            ) {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'avg_pool2d\' requires a tensor or scalar value",
        )),
    }
}

fn avg_pool2d_tensor(
    tensor: &Tensor,
    k: usize,
    s: usize,
    out_shape: Vec<usize>,
) -> Result<Tensor, String> {
    let r = tensor.rank();
    let h = tensor.shape[r - 2];
    let w = tensor.shape[r - 1];

    let out_h = out_shape[r - 2];
    let out_w = out_shape[r - 1];

    let batch_size: usize = tensor.shape[0..r - 2].iter().product();
    let vals = tensor.to_vec_f32();

    let out_plane_size = out_h * out_w;
    let total_out = batch_size * out_plane_size;
    let in_plane_size = h * w;
    let inv_k2 = 1.0 / (k * k) as f32;

    let mut out_vals = vec![0.0f32; total_out];

    out_vals
        .par_chunks_mut(out_plane_size)
        .enumerate()
        .for_each(|(b, out_plane)| {
            let in_plane_offset = b * in_plane_size;
            for oh in 0..out_h {
                let ih_start = oh * s;
                for ow in 0..out_w {
                    let iw_start = ow * s;

                    let mut sum = 0.0f32;
                    for kh in 0..k {
                        let ih = ih_start + kh;
                        for kw in 0..k {
                            let iw = iw_start + kw;
                            let idx = in_plane_offset + ih * w + iw;
                            sum += vals[idx];
                        }
                    }
                    out_plane[oh * out_w + ow] = sum * inv_k2;
                }
            }
        });

    Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
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
    fn test_avg_pool2d_action() {
        let data: Vec<f32> = vec![
            1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0,
        ];
        let tensor = Tensor::from_f32_shape(&data, vec![4, 4]).unwrap();

        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            // (1+3+2+4)/4 = 2.5, (5+7+6+8)/4 = 6.5
            assert_eq!(out.as_f32_slice().unwrap(), &[2.5, 6.5, 2.5, 6.5]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
