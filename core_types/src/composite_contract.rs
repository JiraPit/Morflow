//! Dimensional rules for ordered tensor inputs. Native kernels use the same rules.
use crate::{
    contract::{self, Error, Result},
    DataType, Dimension, Shape, ValueShape, ValueShapeResult,
};

pub fn finish(result: Result<ValueShape>) -> ValueShapeResult {
    match result {
        Ok(shape) => ValueShapeResult::Ok(shape),
        Err(Error::Unknown) => ValueShapeResult::Unknown,
        Err(Error::Invalid(reason)) => ValueShapeResult::Invalid(reason),
    }
}
pub fn tensor(input: &ValueShape) -> Result<&Shape> {
    match input {
        ValueShape::Unknown => Err(Error::Unknown),
        ValueShape::Leaf { kind, shape } if *kind == DataType::Tensor => {
            shape.as_ref().into_option().ok_or(Error::Unknown)
        }
        _ => Err("Expected a Tensor component".into()),
    }
}
pub fn pair(input: &ValueShape) -> Result<(&Shape, &Shape)> {
    match input {
        ValueShape::Composite(items) => {
            if items.len() != 2 {
                return Err(
                    format!("Expected exactly 2 tensor components, got {}", items.len()).into(),
                );
            }
            Ok((tensor(&input[0])?, tensor(&input[1])?))
        }
        ValueShape::Unknown => Err(Error::Unknown),
        ValueShape::Leaf { kind, .. } if *kind == DataType::Composite => Err(Error::Unknown),
        _ => {
            let shape = tensor(input)?;
            Ok((shape, shape))
        }
    }
}
pub fn elements(shape: &Shape) -> Result<Dimension> {
    shape
        .dims()
        .iter()
        .try_fold(Dimension::Known(1), |n, d| n.checked_mul(*d))
        .ok_or(Error::from("Tensor element count overflows"))
}
pub fn output<D: Into<Dimension>>(dims: impl IntoIterator<Item = D>) -> Result<ValueShape> {
    Ok(ValueShape::tensor(contract::shape(dims)?))
}

/// Standard trailing-axis batch broadcasting, including real zero lengths.
pub fn matmul_shape(a: &Shape, b: &Shape, _wildcards: bool) -> Result<Shape> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err("matmul requires tensor ranks at least 2".into());
    }
    let ad = a.dims();
    let bd = b.dims();
    let (ka, kb) = (ad[a.rank() - 1], bd[b.rank() - 2]);
    if !ka.compatible(kb) {
        return Err(format!("Matrix inner dimensions mismatch: {ka} vs {kb}").into());
    }
    let ab = &ad[..a.rank() - 2];
    let bb = &bd[..b.rank() - 2];
    let rank = ab.len().max(bb.len());
    let mut dims = Vec::with_capacity(rank + 2);
    for axis in 0..rank {
        let get = |batch: &[Dimension]| {
            if axis + batch.len() >= rank {
                batch[axis + batch.len() - rank]
            } else {
                Dimension::Known(1)
            }
        };
        let (x, y) = (get(ab), get(bb));
        let d = if x == y || y == 1 {
            x
        } else if x == 1 || x.is_unknown() {
            y
        } else if y.is_unknown() {
            x
        } else {
            return Err(format!("Matrix batch dimensions cannot broadcast: {x} vs {y}").into());
        };
        dims.push(d);
    }
    dims.extend([ad[a.rank() - 2], bd[b.rank() - 1]]);
    contract::shape(dims)
}
pub fn dot(
    input: ValueShape,
    _args: impl Into<crate::shapecheck::PreparedArgs>,
) -> ValueShapeResult {
    finish((|| {
        let (a, b) = pair(&input)?;
        let (na, nb) = (elements(a)?, elements(b)?);
        if !na.compatible(nb) {
            return Err(format!("Vector length mismatch: {na} vs {nb}").into());
        }
        output([1])
    })())
}
pub fn outer(
    input: ValueShape,
    _args: impl Into<crate::shapecheck::PreparedArgs>,
) -> ValueShapeResult {
    finish((|| {
        let (a, b) = pair(&input)?;
        output([elements(a)?, elements(b)?])
    })())
}
pub fn matmul(
    input: ValueShape,
    _args: impl Into<crate::shapecheck::PreparedArgs>,
) -> ValueShapeResult {
    finish((|| {
        let (a, b) = pair(&input)?;
        Ok(ValueShape::tensor(matmul_shape(a, b, true)?))
    })())
}
pub fn cosine_similarity(
    input: ValueShape,
    _args: impl Into<crate::shapecheck::PreparedArgs>,
) -> ValueShapeResult {
    finish((|| {
        if matches!(&input, ValueShape::Leaf {kind,..} if *kind==DataType::Composite) {
            return Err(Error::Unknown);
        }
        if !matches!(input, ValueShape::Composite(_)) {
            return Ok(ValueShape::tensor(tensor(&input)?.clone()));
        }
        let (a, b) = pair(&input)?;
        if a.rank() != b.rank()
            || a.dims()
                .iter()
                .zip(b.dims())
                .any(|(a, b)| !a.compatible(*b))
        {
            return Err("Shape mismatch in cosine_similarity".into());
        }
        output(if a.rank() > 1 {
            a.dims()[..a.rank() - 1].to_vec()
        } else {
            vec![Dimension::Known(1)]
        })
    })())
}
pub fn concat(input: ValueShape, args: impl crate::shapecheck::ArgumentSource) -> ValueShapeResult {
    finish((|| {
        let items = match &input {
            ValueShape::Composite(items) if items.is_empty() => {
                return Err("Cannot concatenate an empty Composite".into())
            }
            ValueShape::Composite(items) => items.as_slice(),
            ValueShape::Unknown => return Err(Error::Unknown),
            ValueShape::Leaf { kind, .. } if *kind == DataType::Composite => {
                return Err(Error::Unknown)
            }
            _ => return Ok(ValueShape::tensor(tensor(&input)?.clone())),
        };
        let first = tensor(&items[0])?;
        if items.len() == 1 {
            return Ok(ValueShape::tensor(first.clone()));
        }
        let raw = contract::arg::<isize>(&args, &["axis", "dim"], Some(0), Some(0))?.unwrap();
        let axis = contract::axis(raw, first.rank(), false)?;
        let mut out = first.dims().to_vec();
        let mut total = Dimension::Known(0);
        for item in items {
            let shape = tensor(item)?;
            if shape.rank() != first.rank() {
                return Err("Concat components must have identical ranks".into());
            }
            for (index, dim) in shape.dims().iter().enumerate() {
                if index != axis && !dim.compatible(out[index]) {
                    return Err(format!("Concat dimension mismatch at axis {index}").into());
                }
                if index != axis && out[index].is_unknown() {
                    out[index] = *dim;
                }
            }
            total = total
                .checked_add(shape.dims()[axis])
                .ok_or(Error::from("Concat axis length overflows"))?;
        }
        out[axis] = total;
        output(out)
    })())
}

pub fn qr(
    input: ValueShape,
    _args: impl Into<crate::shapecheck::PreparedArgs>,
) -> ValueShapeResult {
    finish((|| {
        let shape = tensor(&input)?;
        if shape.rank() != 2 {
            return Err("qr requires a rank-2 Tensor".into());
        }
        let n = shape.dims()[1];
        let q = output(shape.dims().iter().copied())?;
        let r = output([n, n])?;
        Ok(ValueShape::composite([q, r]))
    })())
}
