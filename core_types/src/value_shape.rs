//! Ordered, recursive payload-shape contracts shared by native actions and the checker.
use crate::{ActionArgs, DataType, Payload, RString, RVec, Shape, StableAbi};
use abi_stable::std_types::ROption;
use std::ops::Index;

#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub enum ValueShape {
    Unknown,
    /// `None` means unknown rank; `Known(0)` is an empty dimension.
    Leaf {
        kind: DataType,
        shape: ROption<Shape>,
    },
    /// Components retain exactly the order of the corresponding payload list.
    Composite(RVec<ValueShape>),
}

impl ValueShape {
    pub fn tensor(shape: Shape) -> Self {
        Self::Leaf {
            kind: DataType::Tensor,
            shape: ROption::RSome(shape),
        }
    }
    pub fn unranked(kind: DataType) -> Self {
        Self::Leaf {
            kind,
            shape: ROption::RNone,
        }
    }
    pub fn composite(items: impl IntoIterator<Item = ValueShape>) -> Self {
        Self::Composite(items.into_iter().collect())
    }
    pub fn get(&self, index: usize) -> Option<&Self> {
        self.components()?.get(index)
    }
    pub fn components(&self) -> Option<&[Self]> {
        match self {
            Self::Composite(items) => Some(items.as_slice()),
            _ => None,
        }
    }
    pub fn shape(&self) -> Option<&Shape> {
        match self {
            Self::Leaf {
                shape: ROption::RSome(shape),
                ..
            } => Some(shape),
            _ => None,
        }
    }
    pub fn from_ptype(ty: &crate::PType) -> Self {
        match ty {
            crate::PType::Unknown => Self::Unknown,
            crate::PType::CompositeItems(items) => {
                Self::composite(items.iter().map(Self::from_ptype))
            }
            other => {
                let shape = other.spec().and_then(|spec| spec.dims()).map(|dims| {
                    Shape::new(dims.iter().map(|dim| {
                        dim.fixed()
                            .map_or(crate::Dimension::Unknown, crate::Dimension::Known)
                    }))
                });
                Self::Leaf {
                    kind: other.data_type(),
                    shape: shape.into(),
                }
            }
        }
    }
    pub fn to_ptype(&self) -> crate::PType {
        match self {
            Self::Unknown => crate::PType::Unknown,
            Self::Composite(items) => {
                crate::PType::CompositeItems(items.iter().map(Self::to_ptype).collect())
            }
            Self::Leaf { kind, shape } => {
                let spec = match shape {
                    ROption::RSome(shape) => crate::ShapeSpec::Ranked {
                        dims: shape
                            .dims()
                            .iter()
                            .map(|d| match d {
                                crate::Dimension::Known(n) => crate::Dim::Fixed(*n),
                                crate::Dimension::Unknown => crate::Dim::Any,
                            })
                            .collect(),
                    },
                    ROption::RNone => crate::ShapeSpec::AnyRank,
                };
                if *kind == DataType::Tensor {
                    crate::PType::Tensor(spec)
                } else if *kind == DataType::Image {
                    crate::PType::Image(spec)
                } else if *kind == DataType::Audio {
                    crate::PType::Audio(spec)
                } else {
                    crate::PType::from_data_type(*kind)
                }
            }
        }
    }
    pub fn from_payload(payload: &Payload) -> Self {
        let leaf = |kind, dims: &[usize]| Self::Leaf {
            kind,
            shape: ROption::RSome(Shape::new(dims.iter().copied())),
        };
        match payload.unwrap_payload() {
            Payload::Tensor(t) => leaf(DataType::Tensor, &t.shape),
            Payload::Scalar(t) => leaf(DataType::Scalar, &t.shape),
            Payload::Image(i) => leaf(DataType::Image, &i.tensor.shape),
            Payload::Audio(a) => leaf(DataType::Audio, &a.tensor.shape),
            Payload::Composite(items) => Self::composite(items.iter().map(Self::from_payload)),
            Payload::Data { .. } => Self::unranked(DataType::Bytes),
            Payload::Arg(_) => Self::unranked(DataType(0)),
            _ => Self::Unknown,
        }
    }
    pub fn verify(&self, payload: &Payload) -> Result<(), RString> {
        match (self, payload.unwrap_payload()) {
            (Self::Unknown, _) => Ok(()),
            (Self::Composite(expected), Payload::Composite(actual)) => {
                if expected.len() != actual.len() {
                    return Err(format!(
                        "Expected {} components, produced {}",
                        expected.len(),
                        actual.len()
                    )
                    .into());
                }
                for (index, (shape, item)) in expected.iter().zip(actual).enumerate() {
                    shape
                        .verify(item)
                        .map_err(|e| RString::from(format!("Component {index}: {e}")))?;
                }
                Ok(())
            }
            (Self::Composite(_), _) => Err("Expected Composite payload".into()),
            (Self::Leaf { kind, shape }, _) => {
                let actual = Self::from_payload(payload);
                let Self::Leaf {
                    kind: actual_kind,
                    shape: actual_shape,
                } = actual
                else {
                    return Err("Expected non-Composite payload".into());
                };
                if !kind.intersects(actual_kind) {
                    return Err(format!("Expected {kind}, produced {actual_kind}").into());
                }
                if let (ROption::RSome(expected), ROption::RSome(actual)) = (shape, actual_shape) {
                    if expected.rank() != actual.rank()
                        || expected
                            .dims()
                            .iter()
                            .zip(actual.dims())
                            .any(|(e, a)| !e.compatible(*a))
                    {
                        return Err(format!("Expected shape {expected}, produced {actual}").into());
                    }
                }
                Ok(())
            }
        }
    }
}
impl Index<usize> for ValueShape {
    type Output = ValueShape;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .expect("Composite shape index out of bounds")
    }
}

#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub enum ValueShapeResult {
    Ok(ValueShape),
    Unknown,
    Invalid(RString),
}
pub type GetValueShapeFn = extern "C" fn(ValueShape, ActionArgs) -> ValueShapeResult;

/// Validate the full input contract before executing, then the ordered output tree.
pub fn run(
    payload: Payload,
    contract: GetValueShapeFn,
    process: fn(Payload) -> Payload,
) -> Payload {
    let expected = match contract(
        ValueShape::from_payload(&payload),
        payload.args().cloned().unwrap_or_default(),
    ) {
        ValueShapeResult::Invalid(reason) => return Payload::Error(reason),
        ValueShapeResult::Ok(shape) => Some(shape),
        ValueShapeResult::Unknown => None,
    };
    let output = process(payload);
    if matches!(output, Payload::Error(_)) {
        return output;
    }
    if let Some(expected) = expected {
        if let Err(reason) = expected.verify(&output) {
            return Payload::Error(format!("Action shape contract mismatch: {reason}").into());
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn composite_shapes_preserve_order_and_support_nested_indexing() {
        let tree = ValueShape::composite([
            ValueShape::tensor(Shape::new([2, 3])),
            ValueShape::composite([
                ValueShape::tensor(Shape::new([3, 4])),
                ValueShape::unranked(DataType::Bytes),
            ]),
        ]);
        assert_eq!(tree[0].shape().unwrap().dims(), [2, 3]);
        assert_eq!(tree[1][0].shape().unwrap().dims(), [3, 4]);
        assert!(tree.get(2).is_none());
        assert!(tree[0].get(0).is_none());
        assert_eq!(ValueShape::from_ptype(&tree.to_ptype()), tree);
    }
    #[test]
    fn recursive_contracts_reject_wrong_output_order_and_kind() {
        let expected = ValueShape::composite([
            ValueShape::tensor(Shape::new([2])),
            ValueShape::tensor(Shape::new([3])),
        ]);
        let tensor = |n| Payload::Tensor(crate::Tensor::from_f32_slice(&vec![1.; n]));
        assert!(expected
            .verify(&Payload::Composite(vec![tensor(2), tensor(3)].into()))
            .is_ok());
        assert!(expected
            .verify(&Payload::Composite(vec![tensor(3), tensor(2)].into()))
            .is_err());
        assert!(expected
            .verify(&Payload::Composite(
                vec![
                    tensor(2),
                    Payload::Data {
                        buffer: RVec::new()
                    }
                ]
                .into()
            ))
            .is_err());
    }
}
