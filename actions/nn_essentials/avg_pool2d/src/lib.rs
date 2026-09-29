use core_types::{DataType, Payload, Tensor};
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
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut kernel_size = 2usize;
    let mut stride = 2usize;

    if let Some(args) = &args_opt {
        if let Some(k_str) = args
            .get_named("kernel_size")
            .or_else(|| args.get_named("kernel"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(k) = k_str.parse::<usize>() {
                kernel_size = k.max(1);
            }
        }
        if let Some(s_str) = args
            .get_named("stride")
            .or_else(|| args.positional.get(1).map(|s| s.as_str()))
        {
            if let Ok(s) = s_str.parse::<usize>() {
                stride = s.max(1);
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) => match avg_pool2d_tensor(&tensor, kernel_size, stride) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        Payload::Image(image) => match avg_pool2d_tensor(&image.tensor, kernel_size, stride) {
            Ok(t) => Payload::Tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn avg_pool2d_tensor(tensor: &Tensor, k: usize, s: usize) -> Result<Tensor, String> {
    let r = tensor.rank();
    if r < 2 {
        return Err("avg_pool2d requires tensor of rank at least 2 [H, W]".into());
    }

    let h = tensor.shape[r - 2];
    let w = tensor.shape[r - 1];

    let out_h = if h >= k { (h - k) / s + 1 } else { 0 };
    let out_w = if w >= k { (w - k) / s + 1 } else { 0 };

    if out_h == 0 || out_w == 0 {
        return Err(format!(
            "Output spatial dimensions {}x{} are zero for input {}x{} with kernel {}",
            out_h, out_w, h, w, k
        ));
    }

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

    let mut out_shape = tensor.shape[0..r - 2].to_vec();
    out_shape.push(out_h);
    out_shape.push(out_w);

    Tensor::from_f32_vec(out_vals, out_shape).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
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
