//! Shared validation for native action shape contracts.
use crate::shapecheck::{ArgumentSource, CachedParse};
#[cfg(test)]
use crate::ActionArgs;
use crate::{GetShapeFn, Payload, RString, Shape, ShapeResult};

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
impl From<RString> for Error {
    fn from(s: RString) -> Self {
        Self::Invalid(s)
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
pub fn shape<D: Into<crate::Dimension>>(dims: impl IntoIterator<Item = D>) -> Result<Shape> {
    let shape = Shape::new(dims);
    shape
        .dims()
        .iter()
        .try_fold(crate::Dimension::Known(1), |n, d| n.checked_mul(*d))
        .ok_or(Error::from("Shape element count overflows"))?;
    Ok(shape)
}
pub fn value<'a>(
    args: &'a impl ArgumentSource,
    keys: &[&str],
    position: Option<usize>,
) -> Result<Option<&'a str>> {
    let value = keys
        .iter()
        .find_map(|k| args.raw().get_named(k))
        .or_else(|| position.and_then(|p| args.raw().positional.get(p).map(|s| s.as_str())));
    if value.is_some_and(|v| v.starts_with('$')) {
        return Err(Error::Unknown);
    }
    Ok(value)
}
pub fn arg<T: CachedParse>(
    args: &impl ArgumentSource,
    keys: &[&str],
    position: Option<usize>,
    default: Option<T>,
) -> Result<Option<T>> {
    match value(args, keys, position)? {
        Some(v) => args
            .parse(v)
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
/// Reject dimensions that the image kernels cannot consume.
pub fn image_nonempty(input: &Shape) -> Result<()> {
    if input.dims().iter().any(|d| *d == 0) {
        return Err("Image operations require nonempty spatial dimensions".into());
    }
    Ok(())
}
/// Whether partial dimensions permit both HWC and CHW layout detection.
pub fn image_layout_unknown(input: &Shape) -> bool {
    match input.dims() {
        [a, _, c] => (c.is_unknown() && !(*a > 4)) || (*c > 4 && a.is_unknown()),
        _ => false,
    }
}
/// Uses the same HWC-first layout convention as raw tensor image actions.
pub fn image_dims(
    input: &Shape,
) -> Result<(crate::Dimension, crate::Dimension, crate::Dimension, bool)> {
    image_nonempty(input)?;
    let d = input.dims();
    match d {
        [h, w] => Ok((*h, *w, 1.into(), false)),
        [a, b, c] => {
            if c.is_unknown() && *a > 4 {
                return Ok((*a, *b, *c, false));
            }
            if c.is_unknown() || (*c > 4 && a.is_unknown()) {
                return Ok((
                    crate::Dimension::Unknown,
                    crate::Dimension::Unknown,
                    crate::Dimension::Unknown,
                    false,
                ));
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
pub fn image_shape(
    h: crate::Dimension,
    w: crate::Dimension,
    c: crate::Dimension,
    rank: usize,
    chw: bool,
) -> Result<Shape> {
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
                .any(|(e, a)| !e.matches(*a))
        {
            return Payload::Error(
                format!("Action shape contract mismatch: expected {expected}, produced {actual:?}")
                    .into(),
            );
        }
    }
    output
}

/// Validate a Composite-producing action's component count, kinds, and shapes.
pub fn run_composite(
    payload: Payload,
    contract: GetShapeFn,
    components: crate::GetComponentsFn,
    process: fn(Payload) -> Payload,
) -> Payload {
    let expected = payload_dims(&payload).map(|dims| {
        components(
            Shape::new(dims.iter().copied()),
            payload.args().cloned().unwrap_or_default(),
        )
    });
    let output = run(payload, contract, process);
    if matches!(output, Payload::Error(_)) {
        return output;
    }
    if let Some(expected) = expected {
        let Payload::Composite(items) = output.unwrap_payload() else {
            return Payload::Error("Action component contract requires Composite output".into());
        };
        if items.len() != expected.len() {
            return Payload::Error(
                format!(
                    "Action component contract expected {} components, produced {}",
                    expected.len(),
                    items.len()
                )
                .into(),
            );
        }
        for (index, (component, item)) in expected.iter().zip(items).enumerate() {
            let mut ty = crate::PType::from_data_type(component.kind);
            match &component.shape {
                ShapeResult::Ok(shape) => {
                    ty = ty.with_shape(shape);
                }
                ShapeResult::Invalid(reason) => return Payload::Error(reason.clone()),
                ShapeResult::Unknown => {}
            }
            if let Err(reason) = ty.verify_payload(item) {
                return Payload::Error(
                    format!("Action component {index} contract mismatch: {reason}").into(),
                );
            }
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
    fn known_zero_is_verified_while_unknown_lengths_are_permitted() {
        extern "C" fn empty(_: Shape, _: ActionArgs) -> ShapeResult {
            ShapeResult::Ok(Shape::new([0]))
        }
        extern "C" fn dynamic(_: Shape, _: ActionArgs) -> ShapeResult {
            ShapeResult::Ok(Shape::unknown(1))
        }
        let input = || Payload::Tensor(Tensor::from_f32_slice(&[1.]));
        assert!(matches!(run(input(), empty, three), Payload::Error(_)));
        assert!(matches!(run(input(), dynamic, three), Payload::Tensor(_)));
    }
    #[test]
    fn composite_guard_checks_component_shapes_and_count() {
        extern "C" fn unknown(_: Shape, _: ActionArgs) -> ShapeResult {
            ShapeResult::Unknown
        }
        extern "C" fn components(_: Shape, _: ActionArgs) -> crate::RVec<crate::OutputComponent> {
            vec![crate::OutputComponent {
                kind: crate::DataType::Tensor,
                shape: ShapeResult::Ok(Shape::new([2])),
            }]
            .into()
        }
        fn wrong_shape(_: Payload) -> Payload {
            Payload::Composite(vec![three(Payload::scalar_f32(1.))].into())
        }
        fn wrong_count(_: Payload) -> Payload {
            Payload::Composite(crate::RVec::new())
        }
        for process in [wrong_shape as fn(Payload) -> Payload, wrong_count] {
            assert!(matches!(
                run_composite(Payload::scalar_f32(1.), unknown, components, process),
                Payload::Error(_)
            ));
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
