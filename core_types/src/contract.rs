//! Shared validation for native action shape contracts.
use crate::{ActionArgs, GetShapeFn, Payload, RString, Shape, ShapeResult};
use std::str::FromStr;

#[derive(Debug)]
pub enum Error {
    Unknown,
    Invalid(RString),
}
impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self::Invalid(s.into())
    }
}
impl From<String> for Error {
    fn from(s: String) -> Self {
        Self::Invalid(s.into())
    }
}
pub type Result<T> = std::result::Result<T, Error>;

pub fn finish(result: Result<Shape>) -> ShapeResult {
    match result {
        Ok(shape) => ShapeResult::Ok(shape),
        Err(Error::Unknown) => ShapeResult::Unknown,
        Err(Error::Invalid(reason)) => ShapeResult::Invalid(reason),
    }
}
pub fn shape(dims: impl IntoIterator<Item = usize>) -> Result<Shape> {
    let shape = Shape::new(dims);
    shape
        .dims()
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or(Error::from("Shape element count overflows"))?;
    Ok(shape)
}
pub fn value<'a>(
    args: &'a ActionArgs,
    keys: &[&str],
    position: Option<usize>,
) -> Result<Option<&'a str>> {
    let value = keys
        .iter()
        .find_map(|k| args.get_named(k))
        .or_else(|| position.and_then(|p| args.positional.get(p).map(|s| s.as_str())));
    if value.is_some_and(|v| v.starts_with('$')) {
        return Err(Error::Unknown);
    }
    Ok(value)
}
pub fn arg<T: FromStr>(
    args: &ActionArgs,
    keys: &[&str],
    position: Option<usize>,
    default: Option<T>,
) -> Result<Option<T>> {
    match value(args, keys, position)? {
        Some(v) => v
            .parse()
            .map(Some)
            .map_err(|_| Error::from(format!("Invalid {} argument '{v}'", keys[0]))),
        None => Ok(default),
    }
}
pub fn axis(dim: isize, rank: usize, insertion: bool) -> Result<usize> {
    let limit = rank + usize::from(insertion);
    let resolved = if dim < 0 {
        dim.checked_add(limit as isize)
    } else {
        Some(dim)
    };
    match resolved {
        Some(i) if i >= 0 && (i as usize) < limit => Ok(i as usize),
        _ => Err(format!("Axis {dim} out of bounds for rank {rank}").into()),
    }
}
/// Uses the same HWC-first layout convention as raw tensor image actions.
pub fn image_dims(input: &Shape) -> Result<(usize, usize, usize, bool)> {
    let d = input.dims();
    match d {
        [h, w] => Ok((*h, *w, 1, false)),
        [a, b, c] => {
            if *c == 0 || (*c > 4 && *a == 0) {
                return Err(Error::Unknown);
            }
            if *c > 4 && *a <= 4 {
                Ok((*b, *c, *a, true))
            } else {
                Ok((*a, *b, *c, false))
            }
        }
        _ => Err("Image operations require rank 2 or 3 tensors".into()),
    }
}
pub fn image_shape(h: usize, w: usize, c: usize, rank: usize, chw: bool) -> Result<Shape> {
    if rank == 2 {
        shape([h, w])
    } else if chw {
        shape([c, h, w])
    } else {
        shape([h, w, c])
    }
}
/// Image kernels support F32/U8 and consume contiguous pixel buffers. Crop can
/// retain a strided view, so callers choose whether normalization is required.
pub fn image_input(payload: Payload, contiguous: bool) -> std::result::Result<Payload, RString> {
    let (inner, args) = payload.take_payload_and_args();
    let inner = match inner {
        Payload::Tensor(tensor) => {
            if !matches!(
                tensor.dtype,
                crate::TensorDType::F32 | crate::TensorDType::U8
            ) {
                return Err("Image actions require F32 or U8 tensors".into());
            }
            if tensor.shape.contains(&0) {
                return Err("Image operations require nonempty spatial dimensions".into());
            }
            let tensor = if contiguous && !tensor.is_contiguous() {
                crate::Tensor::from_rvec_u8(
                    tensor.to_contiguous_bytes(),
                    tensor.shape.to_vec(),
                    tensor.dtype,
                )?
            } else {
                tensor
            };
            Payload::Tensor(tensor)
        }
        other => other,
    };
    Ok(match args {
        Some(args) => Payload::WithArgs {
            payload: crate::RBox::new(inner),
            args,
        },
        None => inner,
    })
}

fn payload_dims(payload: &Payload) -> Option<&[usize]> {
    match payload.unwrap_payload() {
        Payload::Tensor(t) | Payload::Scalar(t) => Some(t.shape.as_slice()),
        Payload::Image(i) => Some(i.tensor.shape.as_slice()),
        Payload::Audio(a) => Some(a.tensor.shape.as_slice()),
        _ => None,
    }
}
/// Checks a native action's arguments before execution and its claimed output
/// shape afterwards. Composite and opaque byte inputs have no single shape.
pub fn run(payload: Payload, contract: GetShapeFn, process: fn(Payload) -> Payload) -> Payload {
    let verdict = payload_dims(&payload).map(|dims| {
        contract(
            Shape::new(dims.iter().copied()),
            payload.args().cloned().unwrap_or_default(),
        )
    });
    if let Some(ShapeResult::Invalid(reason)) = &verdict {
        return Payload::Error(reason.clone());
    }
    let output = process(payload);
    if let (Some(ShapeResult::Ok(expected)), Some(actual)) = (verdict, payload_dims(&output)) {
        if expected.rank() != actual.len()
            || expected
                .dims()
                .iter()
                .zip(actual)
                .any(|(e, a)| *e != 0 && e != a)
        {
            return Payload::Error(
                format!("Action shape contract mismatch: expected {expected}, produced {actual:?}")
                    .into(),
            );
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    extern "C" fn invalid(_: Shape, _: ActionArgs) -> ShapeResult {
        ShapeResult::Invalid("invalid call".into())
    }
    extern "C" fn wrong(_: Shape, _: ActionArgs) -> ShapeResult {
        ShapeResult::Ok(Shape::new([2]))
    }
    fn never(_: Payload) -> Payload {
        panic!("Invalid calls must be rejected before processing")
    }
    fn three(_: Payload) -> Payload {
        Payload::Tensor(Tensor::from_f32_slice(&[1.0, 2.0, 3.0]))
    }
    #[test]
    fn native_guard_rejects_invalid_calls_and_incorrect_output_contracts() {
        let input = || Payload::Tensor(Tensor::from_f32_slice(&[1.0]));
        assert!(matches!(run(input(), invalid, never), Payload::Error(_)));
        match run(input(), wrong, three) {
            Payload::Error(reason) => assert!(reason.contains("contract mismatch")),
            other => panic!("Expected error, got {other:?}"),
        }
    }
    #[test]
    fn shared_helpers_reject_overflow_and_unresolved_shape_arguments() {
        assert!(matches!(shape([usize::MAX, 2]), Err(Error::Invalid(_))));
        let args = ActionArgs {
            positional: vec!["$axis".into()].into(),
            named: crate::RVec::new(),
        };
        assert!(matches!(
            arg::<usize>(&args, &["axis"], Some(0), Some(0)),
            Err(Error::Unknown)
        ));
    }
}
