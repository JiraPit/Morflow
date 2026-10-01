use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};
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
    if !matches!(input.rank(), 1 | 2) {
        return ShapeResult::Invalid("diag requires rank 1 or 2".into());
    }
    let rank = if input.rank() == 1 { 2 } else { 1 };
    let result = contract::finish((|| {
        let k = arg::<isize>(&args, &["diagonal", "k"], Some(0), Some(0))?.unwrap();
        let offset = k.unsigned_abs();
        match input.dims() {
            [n] => {
                let size = n
                    .checked_add(offset)
                    .ok_or(Error::from("Diagonal matrix dimension overflows"))?;
                contract::shape([size, size])
            }
            [h, w] => {
                let len = if k >= 0 {
                    (*h).min(w.saturating_sub(offset))
                } else {
                    h.saturating_sub(offset).min(*w)
                };
                contract::shape([len])
            }
            _ => Err("diag requires rank 1 or 2".into()),
        }
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
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut k = 0isize;
    if let Some(args) = &args_opt {
        if let Some(k_str) = args
            .get_named("diagonal")
            .or_else(|| args.get_named("k"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(val) = prepared.args.parse::<isize>(k_str) {
                k = val;
            }
        }
    }

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match compute_diag(&tensor, k) {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_diag(tensor: &Tensor, k: isize) -> Result<Tensor, String> {
    let r = tensor.rank();
    let vals = tensor.to_vec_f32();

    if r == 1 {
        // 1D -> 2D diagonal matrix
        let n = vals.len();
        let k_abs = k.unsigned_abs();
        let size = n + k_abs;
        let mut mat = vec![0.0f32; size * size];

        for (i, &val) in vals.iter().enumerate() {
            let row = if k >= 0 { i } else { i + k_abs };
            let col = if k >= 0 { i + k_abs } else { i };
            mat[row * size + col] = val;
        }

        Tensor::from_f32_vec(mat, vec![size, size]).map_err(|e| e.to_string())
    } else if r == 2 {
        // 2D -> 1D diagonal vector
        let h = tensor.shape[0];
        let w = tensor.shape[1];

        let diag_len = if k >= 0 {
            if (k as usize) < w {
                h.min(w - k as usize)
            } else {
                0
            }
        } else {
            let k_pos = k.unsigned_abs();
            if k_pos < h {
                (h - k_pos).min(w)
            } else {
                0
            }
        };

        let mut diag = vec![0.0f32; diag_len];
        for (i, slot) in diag.iter_mut().enumerate() {
            let row = if k >= 0 { i } else { i + k.unsigned_abs() };
            let col = if k >= 0 { i + k as usize } else { i };
            *slot = vals[row * w + col];
        }

        Tensor::from_f32_vec(diag, vec![diag_len]).map_err(|e| e.to_string())
    } else {
        Err(format!("diag requires 1D or 2D tensor, got rank {}", r))
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
    use core_types::Tensor;

    #[test]
    fn test_diag_action_2d_to_1d() {
        let mat = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let res = process(Payload::Tensor(mat));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[1.0, 4.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_diag_action_1d_to_2d() {
        let vec = Tensor::from_f32_slice(&[5.0, 6.0]);
        let res = process(Payload::Tensor(vec));
        if let Payload::Tensor(out) = res {
            assert_eq!(out.shape.as_slice(), &[2, 2]);
            assert_eq!(out.as_f32_slice().unwrap(), &[5.0, 0.0, 0.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
