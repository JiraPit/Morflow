//! Concrete shapes, shape specifications, and the static pipeline type model.
//!
//! Three related concepts live here:
//!
//! * [`Shape`] — a known-rank, possibly partial shape, as returned by an action's
//!   `get_output_shape` across the FFI boundary.
//! * [`ShapeSpec`] — what a user can write in a pipeline, i.e. a shape with
//!   wildcards (`Tensor[*,*,3]`, `Tensor[rank=2]`).
//! * [`PType`] — the full declared type of a pipeline value, combining a kind
//!   (`Bytes`, `Scalar`, `Tensor`, `Image`, `Audio`, `Composite`, or one of
//!   the argument kinds) with an optional shape specification.

use crate::{ActionArgs, AudioLayout, DataType, ImageLayout, Payload, RString, RVec, StableAbi};
use std::fmt;

pub const SHAPE_CONTRACT_ABI_VERSION: u32 = 2;

/// An explicit dimension in a native shape contract. Zero is a real length.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    Known(usize),
    Unknown,
}
impl Dimension {
    pub fn known(self) -> Option<usize> {
        match self {
            Self::Known(n) => Some(n),
            Self::Unknown => None,
        }
    }
    pub fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }
    pub fn matches(self, actual: usize) -> bool {
        self.known().is_none_or(|n| n == actual)
    }
    pub fn compatible(self, other: Self) -> bool {
        self.is_unknown() || other.is_unknown() || self == other
    }
    pub fn checked_add(self, other: impl Into<Self>) -> Option<Self> {
        match (self, other.into()) {
            (Self::Known(a), Self::Known(b)) => a.checked_add(b).map(Self::Known),
            _ => Some(Self::Unknown),
        }
    }
    pub fn checked_mul(self, other: impl Into<Self>) -> Option<Self> {
        match (self, other.into()) {
            (Self::Known(0), _) | (_, Self::Known(0)) => Some(Self::Known(0)),
            (Self::Known(a), Self::Known(b)) => a.checked_mul(b).map(Self::Known),
            _ => Some(Self::Unknown),
        }
    }
    pub fn saturating_sub(self, other: usize) -> Self {
        self.known()
            .map_or(Self::Unknown, |n| Self::Known(n.saturating_sub(other)))
    }
    pub fn min(self, other: Self) -> Self {
        match (self, other) {
            (Self::Known(0), _) | (_, Self::Known(0)) => Self::Known(0),
            (Self::Known(a), Self::Known(b)) => Self::Known(a.min(b)),
            _ => Self::Unknown,
        }
    }
}
impl From<usize> for Dimension {
    fn from(n: usize) -> Self {
        Self::Known(n)
    }
}
impl PartialEq<usize> for Dimension {
    fn eq(&self, n: &usize) -> bool {
        *self == Self::Known(*n)
    }
}
impl PartialOrd<usize> for Dimension {
    fn partial_cmp(&self, n: &usize) -> Option<std::cmp::Ordering> {
        self.known().map(|a| a.cmp(n))
    }
}
impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known(n) => write!(f, "{n}"),
            Self::Unknown => write!(f, "*"),
        }
    }
}

/// A known-rank shape whose dimensions may be concrete or unknown.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape {
    pub dims: RVec<Dimension>,
}
impl Shape {
    pub fn new<D: Into<Dimension>>(dims: impl IntoIterator<Item = D>) -> Self {
        Self {
            dims: dims.into_iter().map(Into::into).collect(),
        }
    }
    pub fn unknown(rank: usize) -> Self {
        Self::new(vec![Dimension::Unknown; rank])
    }
    pub fn scalar() -> Self {
        Self { dims: RVec::new() }
    }
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
    pub fn dims(&self) -> &[Dimension] {
        self.dims.as_slice()
    }
    /// Returns concrete lengths only when every dimension is known.
    pub fn known_dims(&self) -> Option<Vec<usize>> {
        self.dims.iter().map(|d| d.known()).collect()
    }
    pub fn element_count(&self) -> Option<usize> {
        let count = self
            .dims
            .iter()
            .try_fold(Dimension::Known(1), |n, d| n.checked_mul(*d))?;
        count.known()
    }
}
impl From<Shape> for ShapeSpec {
    fn from(shape: Shape) -> ShapeSpec {
        Self::from_dimensions(shape.dims())
    }
}
impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.dims
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// A single dimension of a shape specification: a wildcard or an exact length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    /// `*` — any length.
    Any,
    /// An exact length, e.g. the `3` in `Tensor[*,*,3]`.
    Fixed(usize),
}

impl Dim {
    /// Whether this dimension accepts `actual`.
    pub fn matches(&self, actual: usize) -> bool {
        match self {
            Dim::Any => true,
            Dim::Fixed(n) => *n == actual,
        }
    }

    /// Whether this dimension is a wildcard.
    pub fn is_any(&self) -> bool {
        matches!(self, Dim::Any)
    }

    /// The pinned length, if any.
    pub fn fixed(&self) -> Option<usize> {
        match self {
            Dim::Any => None,
            Dim::Fixed(n) => Some(*n),
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Any => write!(f, "*"),
            Dim::Fixed(n) => write!(f, "{}", n),
        }
    }
}

/// A shape as written in a pipeline: an unknown rank, or a fixed rank whose
/// dimensions may contain wildcards.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ShapeSpec {
    /// Rank is not pinned: `Tensor`, `Image`, `Audio`.
    #[default]
    AnyRank,
    /// Rank is pinned: `Tensor[2,3]`, `Image[*,*,3]`, `Tensor[rank=2]`.
    Ranked { dims: Vec<Dim> },
}

impl ShapeSpec {
    /// A pinned rank with every dimension left as a wildcard, i.e. `[rank=n]`.
    pub fn from_rank(rank: usize) -> Self {
        ShapeSpec::Ranked {
            dims: vec![Dim::Any; rank],
        }
    }

    /// A pinned shape from concrete lengths, i.e. `[2,3]`.
    pub fn from_lengths(lengths: &[usize]) -> Self {
        ShapeSpec::Ranked {
            dims: lengths.iter().map(|n| Dim::Fixed(*n)).collect(),
        }
    }

    pub fn from_dimensions(dims: &[Dimension]) -> Self {
        Self::Ranked {
            dims: dims
                .iter()
                .map(|d| match d {
                    Dimension::Known(n) => Dim::Fixed(*n),
                    Dimension::Unknown => Dim::Any,
                })
                .collect(),
        }
    }

    /// Parses a bracketed shape specification, with or without brackets.
    ///
    /// Accepts `[*,*,3]`, `[rank=2]`, `[2,3]`, `rank=2`, and the empty string
    /// (meaning any rank).
    pub fn parse(raw: &str) -> Result<ShapeSpec, String> {
        let trimmed = raw.trim();
        let inner = if trimmed.starts_with('[') && trimmed.ends_with(']') {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };
        let inner = inner.trim();
        if inner.is_empty() {
            return Ok(ShapeSpec::AnyRank);
        }
        if let Some(rank) = inner.strip_prefix("rank=") {
            let rank = rank.trim();
            let parsed = rank
                .parse::<usize>()
                .map_err(|_| format!("'rank={}' is not a valid dimension count", rank))?;
            if parsed == 0 {
                return Err(
                    "rank=0 is not a valid shape; use the 'Scalar' type for rank-0 values"
                        .to_string(),
                );
            }
            return Ok(ShapeSpec::from_rank(parsed));
        }
        let mut dims = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(format!("'{}' is not a valid shape specification", raw));
            }
            if part == "*" {
                dims.push(Dim::Any);
            } else {
                let n = part.parse::<usize>().map_err(|_| {
                    format!(
                        "'{}' is not a valid dimension; expected a number or '*'",
                        part
                    )
                })?;
                dims.push(Dim::Fixed(n));
            }
        }
        if dims.is_empty() {
            return Ok(ShapeSpec::AnyRank);
        }
        Ok(ShapeSpec::Ranked { dims })
    }

    /// The pinned rank, if any.
    pub fn rank(&self) -> Option<usize> {
        match self {
            ShapeSpec::AnyRank => None,
            ShapeSpec::Ranked { dims } => Some(dims.len()),
        }
    }

    /// The dimensions, if the rank is pinned.
    pub fn dims(&self) -> Option<&[Dim]> {
        match self {
            ShapeSpec::AnyRank => None,
            ShapeSpec::Ranked { dims } => Some(dims.as_slice()),
        }
    }

    /// Whether the rank is not pinned.
    pub fn is_any_rank(&self) -> bool {
        matches!(self, ShapeSpec::AnyRank)
    }

    /// Whether every pinned dimension is a wildcard.
    pub fn is_all_any(&self) -> bool {
        match self {
            ShapeSpec::AnyRank => true,
            ShapeSpec::Ranked { dims } => dims.iter().all(Dim::is_any),
        }
    }

    /// Drops the dimension pinned at `axis`, shifting the ones after it down.
    pub fn without_axis(&self, axis: usize) -> ShapeSpec {
        match self {
            ShapeSpec::AnyRank => ShapeSpec::AnyRank,
            ShapeSpec::Ranked { dims } => {
                let mut dims = dims.clone();
                if axis < dims.len() {
                    dims.remove(axis);
                }
                ShapeSpec::Ranked { dims }
            }
        }
    }

    /// Checks `actual` against this specification.
    pub fn matches_lengths(&self, actual: &[usize]) -> bool {
        match self {
            ShapeSpec::AnyRank => true,
            ShapeSpec::Ranked { dims } => {
                dims.len() == actual.len() && dims.iter().zip(actual).all(|(d, a)| d.matches(*a))
            }
        }
    }
}

impl fmt::Display for ShapeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeSpec::AnyRank => Ok(()),
            // All-wildcard ranks are written back in their short form.
            ShapeSpec::Ranked { dims } if dims.iter().all(Dim::is_any) => {
                write!(f, "[rank={}]", dims.len())
            }
            ShapeSpec::Ranked { dims } => {
                let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                write!(f, "[{}]", parts.join(", "))
            }
        }
    }
}

/// A rank-0 shape specification, shared by every scalar value.
static SCALAR_SPEC: ShapeSpec = ShapeSpec::Ranked { dims: Vec::new() };

/// The four argument types, which exist only as pipeline parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    Int,
    Float,
    Str,
    Bool,
}

impl ArgKind {
    /// The keyword used in a pipeline declaration.
    pub fn name(&self) -> &'static str {
        match self {
            ArgKind::Int => "IntArg",
            ArgKind::Float => "FloatArg",
            ArgKind::Str => "StrArg",
            ArgKind::Bool => "BoolArg",
        }
    }

    /// Whether a textual argument value is valid for this kind.
    pub fn accepts(&self, raw: &str) -> bool {
        match self {
            ArgKind::Int => raw.trim().parse::<i64>().is_ok(),
            ArgKind::Float => raw.trim().parse::<f64>().is_ok(),
            ArgKind::Str => true,
            ArgKind::Bool => matches!(raw.trim(), "true" | "false" | "0" | "1"),
        }
    }

    /// How a value of this kind is described in a diagnostic.
    pub fn describe(&self) -> &'static str {
        match self {
            ArgKind::Int => "an integer",
            ArgKind::Float => "a float",
            ArgKind::Str => "a string",
            ArgKind::Bool => "a boolean",
        }
    }
}

impl fmt::Display for ArgKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// The full type of a pipeline value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PType {
    /// Not yet known (only ever used as a checker placeholder).
    #[default]
    Unknown,
    /// An opaque byte buffer, i.e. a `Payload::Data`.
    Bytes,
    /// An `IntArg`, `FloatArg`, `StrArg`, or `BoolArg` parameter. These never
    /// flow between actions; they are only readable as action arguments.
    Arg(ArgKind),
    /// A rank-0 numeric value.
    Scalar,
    /// A tensor, with an optional shape specification.
    Tensor(ShapeSpec),
    /// An image, with an optional shape specification. The channel dimension is
    /// validated against the payload's own layout.
    Image(ShapeSpec),
    /// An audio payload, with an optional shape specification. Channel and
    /// sample dimensions are validated against the payload's own layout.
    Audio(ShapeSpec),
    /// A tuple of payloads.
    Composite,
    /// Composite with known component types, in payload order.
    CompositeItems(Vec<PType>),
}

impl PType {
    /// A value type of `dt` with no shape information.
    ///
    /// `dt` may name several kinds, as a rank-preserving action declares both
    /// `Tensor` and `Scalar`; the shape reported for the actual call then
    /// decides between them.
    pub fn from_data_type(dt: DataType) -> PType {
        if dt.contains(DataType::Tensor) {
            PType::Tensor(ShapeSpec::AnyRank)
        } else if dt.contains(DataType::Image) {
            PType::Image(ShapeSpec::AnyRank)
        } else if dt.contains(DataType::Audio) {
            PType::Audio(ShapeSpec::AnyRank)
        } else if dt.contains(DataType::Composite) {
            PType::Composite
        } else if dt.contains(DataType::Scalar) {
            PType::Scalar
        } else if dt.contains(DataType::Bytes) {
            PType::Bytes
        } else {
            PType::Unknown
        }
    }

    /// The shape specification, for the shape-carrying kinds.
    pub fn spec(&self) -> Option<&ShapeSpec> {
        match self {
            PType::Tensor(spec) | PType::Image(spec) | PType::Audio(spec) => Some(spec),
            PType::Scalar => Some(&SCALAR_SPEC),
            _ => None,
        }
    }

    /// The pinned rank, if this type has a pinned rank.
    pub fn rank(&self) -> Option<usize> {
        self.spec().and_then(|spec| spec.rank())
    }

    /// Whether this is the rank-0 scalar type.
    pub fn is_scalar(&self) -> bool {
        matches!(self, PType::Scalar)
    }

    /// Whether this type is still unknown.
    pub fn is_unknown(&self) -> bool {
        matches!(self, PType::Unknown)
    }

    /// Whether this type is one of the argument types.
    pub fn arg_kind(&self) -> Option<ArgKind> {
        match self {
            PType::Arg(kind) => Some(*kind),
            _ => None,
        }
    }

    /// The `DataType` bits describing which payload kinds this type allows.
    pub fn data_type(&self) -> DataType {
        match self {
            PType::Unknown => DataType::Any,
            PType::Bytes => DataType::Bytes,
            // Arguments are never flowable, so they accept nothing.
            PType::Arg(_) => DataType(0),
            PType::Scalar => DataType::Scalar,
            PType::Tensor(_) => DataType::Tensor,
            PType::Image(_) => DataType::Image,
            PType::Audio(_) => DataType::Audio,
            PType::Composite | PType::CompositeItems(_) => DataType::Composite,
        }
    }

    /// Refines this type with a shape reported by an action.
    ///
    /// A rank-0 result of a tensor-valued action is a `Scalar`; anything else
    /// keeps its kind and takes the produced shape as its specification.
    pub fn with_shape(self, shape: &Shape) -> PType {
        match self {
            PType::Tensor(_) => {
                if shape.is_scalar() {
                    PType::Scalar
                } else {
                    PType::Tensor(ShapeSpec::from_dimensions(shape.dims()))
                }
            }
            PType::Image(_) => PType::Image(ShapeSpec::from_dimensions(shape.dims())),
            PType::Audio(_) => PType::Audio(ShapeSpec::from_dimensions(shape.dims())),
            other => other,
        }
    }

    /// The type of the loop variable when `each` iterates this type.
    ///
    /// Iteration always removes exactly one dimension, so the loop variable is
    /// the same kind one rank lower — except that a rank-1 payload has no
    /// dimension left to iterate and yields `Scalar` values instead.
    pub fn each_loop_var(&self) -> Result<PType, String> {
        match self {
            PType::Tensor(spec) => match spec.rank() {
                Some(0) => Err(format!(
                    "cannot iterate {}: a rank-0 value has no dimension to iterate",
                    self
                )),
                Some(1) => Ok(PType::Scalar),
                Some(_) => Ok(PType::Tensor(spec.without_axis(0))),
                None => Err(format!(
                    "cannot iterate {}: its rank is not known, so the type of the loop variable is ambiguous",
                    self
                )),
            },
            PType::Image(spec) => match spec.rank() {
                Some(0) | Some(1) => Err(format!(
                    "cannot iterate {}: 'each' needs rank >= 2 to iterate one channel at a time",
                    self
                )),
                Some(rank) => Ok(PType::Image(ShapeSpec::from_rank(rank - 1))),
                None => Err(format!(
                    "cannot iterate {}: its rank is not known, so the type of the loop variable is ambiguous",
                    self
                )),
            },
            PType::Audio(spec) => match spec.rank() {
                Some(0) => Err(format!(
                    "cannot iterate {}: a rank-0 value has no dimension to iterate",
                    self
                )),
                // Rank-1 audio is mono, so each iteration is a single sample.
                Some(1) => Ok(PType::Scalar),
                Some(_) => Ok(PType::Audio(ShapeSpec::from_rank(1))),
                None => Err(format!(
                    "cannot iterate {}: its rank is not known, so the type of the loop variable is ambiguous",
                    self
                )),
            },
            other => Err(format!("cannot iterate {}", other)),
        }
    }

    /// Validates a payload supplied by a host against this declared type.
    pub fn verify_payload(&self, payload: &Payload) -> Result<(), String> {
        let inner = payload.unwrap_payload();
        match self {
            PType::Unknown => Ok(()),
            PType::Bytes => match inner {
                Payload::Data { .. } => Ok(()),
                other => Err(kind_mismatch(self, other)),
            },
            PType::Arg(kind) => match inner {
                Payload::Arg(bytes) => {
                    let text = String::from_utf8_lossy(bytes.as_slice());
                    if kind.accepts(&text) {
                        Ok(())
                    } else {
                        Err(format!(
                            "{} expects {} but got '{}'",
                            self,
                            kind.describe(),
                            text
                        ))
                    }
                }
                other => Err(kind_mismatch(self, other)),
            },
            PType::Scalar => match inner {
                Payload::Scalar(t) if t.rank() == 0 => Ok(()),
                Payload::Scalar(t) => Err(format!(
                    "Scalar must be rank 0, but the supplied payload has rank {}",
                    t.rank()
                )),
                // A host is allowed to hand over an untagged rank-0 tensor.
                Payload::Tensor(t) if t.rank() == 0 => Ok(()),
                other => Err(kind_mismatch(self, other)),
            },
            PType::Tensor(spec) => match inner {
                Payload::Tensor(t) => check_shape_spec(spec, t.shape.as_slice(), "Tensor"),
                other => Err(kind_mismatch(self, other)),
            },
            PType::Image(spec) => match inner {
                Payload::Image(img) => {
                    let shape = img.tensor.shape.as_slice();
                    // Channel validation is layout-aware, so compare against the
                    // image's own [height, width, channels] ordering.
                    let hwc = match (shape.len(), img.layout) {
                        (3, ImageLayout::Hwc) => shape.to_vec(),
                        (3, ImageLayout::Chw) => vec![shape[1], shape[2], shape[0]],
                        _ => shape.to_vec(),
                    };
                    check_shape_spec(spec, &hwc, "Image")
                }
                other => Err(kind_mismatch(self, other)),
            },
            PType::Audio(spec) => match inner {
                Payload::Audio(audio) => {
                    let shape = audio.tensor.shape.as_slice();
                    let ordered = match (shape.len(), audio.layout) {
                        (1, _) => shape.to_vec(),
                        (2, AudioLayout::Planar) => shape.to_vec(),
                        // Interleaved is [samples, channels]; the specification
                        // is written in [channels, samples] order.
                        (2, AudioLayout::Interleaved) => vec![shape[1], shape[0]],
                        _ => {
                            return Err(format!(
                                "Audio must be rank 1 or 2, but the supplied payload has rank {}",
                                shape.len()
                            ))
                        }
                    };
                    check_shape_spec(spec, &ordered, "Audio")
                }
                other => Err(kind_mismatch(self, other)),
            },
            PType::CompositeItems(types) => match inner {
                Payload::Composite(items) if items.len() == types.len() => {
                    for (index, (ty, item)) in types.iter().zip(items).enumerate() {
                        ty.verify_payload(item)
                            .map_err(|e| format!("Composite component {index}: {e}"))?;
                    }
                    Ok(())
                }
                Payload::Composite(items) => Err(format!(
                    "Composite expects {} components, got {}",
                    types.len(),
                    items.len()
                )),
                other => Err(kind_mismatch(self, other)),
            },
            PType::Composite => match inner {
                Payload::Composite(_) => Ok(()),
                other => Err(kind_mismatch(self, other)),
            },
        }
    }
}

impl fmt::Display for PType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PType::Unknown => write!(f, "Unknown"),
            PType::Bytes => write!(f, "Bytes"),
            PType::Arg(kind) => write!(f, "{}", kind),
            PType::Scalar => write!(f, "Scalar"),
            PType::Tensor(spec) => write!(f, "Tensor{}", spec),
            PType::Image(spec) => write!(f, "Image{}", spec),
            PType::Audio(spec) => write!(f, "Audio{}", spec),
            PType::Composite => write!(f, "Composite"),
            PType::CompositeItems(items) => write!(
                f,
                "Composite[{}]",
                items
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// The verdict an action returns from its exported shape function.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub enum ShapeResult {
    /// The call is valid and produces this concrete shape.
    Ok(Shape),
    /// Static information is insufficient to determine validity or output shape.
    Unknown,
    /// The arguments make the call invalid (e.g. an out-of-bounds axis).
    Invalid(RString),
}

impl From<Shape> for ShapeResult {
    fn from(shape: Shape) -> ShapeResult {
        ShapeResult::Ok(shape)
    }
}

/// Optional component contract exported by Composite-producing actions.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct OutputComponent {
    pub kind: DataType,
    pub shape: ShapeResult,
}
pub type GetComponentsFn = extern "C" fn(input: Shape, args: ActionArgs) -> RVec<OutputComponent>;

/// Signature of an action's `get_output_shape` export.
pub type GetShapeFn = extern "C" fn(input: Shape, args: ActionArgs) -> ShapeResult;

/// The reason a reducer's `axis` argument misses `rank` dimensions, matching
/// the runtime's axis handling so the checker can repeat the verdict.
pub fn reducer_axis_reason(raw: isize, rank: usize) -> String {
    let valid = if rank == 0 {
        "none (a rank-0 value has no axis to reduce)".to_string()
    } else {
        format!("{} to {}", -(rank as isize), rank as isize - 1)
    };
    format!(
        "axis '{}' is out of bounds for a rank-{} input (valid axes: {})",
        raw, rank, valid
    )
}

fn check_shape_spec(spec: &ShapeSpec, actual: &[usize], label: &str) -> Result<(), String> {
    match spec {
        ShapeSpec::AnyRank => Ok(()),
        ShapeSpec::Ranked { dims } => {
            if dims.len() != actual.len() {
                return Err(format!(
                    "{} expects rank {} (shape {}) but the supplied payload has rank {} (shape [{}])",
                    label,
                    dims.len(),
                    spec,
                    actual.len(),
                    join_usize(actual)
                ));
            }
            for (axis, dim) in dims.iter().enumerate() {
                if let Dim::Fixed(expected) = dim {
                    if actual[axis] != *expected {
                        return Err(format!(
                            "{} expects dimension {} to be {} but the supplied payload has {} (shape [{}])",
                            label,
                            axis,
                            expected,
                            actual[axis],
                            join_usize(actual)
                        ));
                    }
                }
            }
            Ok(())
        }
    }
}

fn kind_mismatch(expected: &PType, actual: &Payload) -> String {
    format!(
        "expected {} but the supplied payload is {}",
        expected,
        crate::payload_kind_name(actual)
    )
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(", ")
}

/// Renders an argument payload's bytes as text.
pub fn arg_text(payload: &Payload) -> Option<String> {
    match payload.unwrap_payload() {
        Payload::Arg(bytes) => Some(String::from_utf8_lossy(bytes.as_slice()).into_owned()),
        _ => None,
    }
}

/// Renders a rank-0 value as a string, honouring the underlying dtype.
pub fn scalar_text(payload: &Payload) -> Result<String, RString> {
    let tensor = match payload.unwrap_payload() {
        Payload::Scalar(t) | Payload::Tensor(t) => t,
        other => {
            return Err(RString::from(format!(
                "expected a scalar value, got {}",
                crate::payload_kind_name(other)
            )))
        }
    };
    if tensor.rank() != 0 {
        return Err(RString::from(format!(
            "expected a scalar value, got a rank-{} tensor",
            tensor.rank()
        )));
    }
    Ok(tensor_scalar_text(tensor))
}

/// Renders a rank-0 tensor's single element according to its dtype.
pub fn tensor_scalar_text(tensor: &crate::Tensor) -> String {
    use crate::TensorDType;
    let bytes = tensor.as_bytes().unwrap_or(&[]);
    let text = |raw: &[u8]| -> String { format!("{:?}", raw) };
    match tensor.dtype {
        TensorDType::F32 => match bytes.get(0..4) {
            Some(b) => format!("{}", f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            None => text(bytes),
        },
        TensorDType::F64 => match bytes.get(0..8) {
            Some(b) => format!(
                "{}",
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            ),
            None => text(bytes),
        },
        TensorDType::I32 => match bytes.get(0..4) {
            Some(b) => format!("{}", i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            None => text(bytes),
        },
        TensorDType::U32 => match bytes.get(0..4) {
            Some(b) => format!("{}", u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            None => text(bytes),
        },
        TensorDType::I64 => match bytes.get(0..8) {
            Some(b) => format!(
                "{}",
                i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            ),
            None => text(bytes),
        },
        TensorDType::U64 => match bytes.get(0..8) {
            Some(b) => format!(
                "{}",
                u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            ),
            None => text(bytes),
        },
        TensorDType::I16 => match bytes.get(0..2) {
            Some(b) => format!("{}", i16::from_le_bytes([b[0], b[1]])),
            None => text(bytes),
        },
        TensorDType::I8 => match bytes.get(0..1) {
            Some(b) => format!("{}", i8::from_le_bytes([b[0]])),
            None => text(bytes),
        },
        TensorDType::U8 => match bytes.get(0..1) {
            Some(b) => format!("{}", u8::from_le_bytes([b[0]])),
            None => text(bytes),
        },
    }
}

/// Reads a rank-0 tensor's single element as a number, honouring its dtype.
pub fn scalar_number(tensor: &crate::Tensor) -> Option<f64> {
    use crate::TensorDType;
    let bytes = tensor.as_bytes()?;
    match tensor.dtype {
        TensorDType::F32 => bytes
            .get(0..4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
        TensorDType::F64 => bytes
            .get(0..8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])),
        TensorDType::I32 => bytes
            .get(0..4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
        TensorDType::U32 => bytes
            .get(0..4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
        TensorDType::I64 => bytes
            .get(0..8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64),
        TensorDType::U64 => bytes
            .get(0..8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64),
        TensorDType::I16 => bytes
            .get(0..2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f64),
        TensorDType::I8 => bytes.get(0..1).map(|b| i8::from_le_bytes([b[0]]) as f64),
        TensorDType::U8 => bytes.get(0..1).map(|b| u8::from_le_bytes([b[0]]) as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rank_sugar() {
        assert_eq!(
            ShapeSpec::parse("[rank=2]").unwrap(),
            ShapeSpec::from_rank(2)
        );
        assert_eq!(ShapeSpec::parse("rank=2").unwrap(), ShapeSpec::from_rank(2));
        assert_eq!(ShapeSpec::parse("").unwrap(), ShapeSpec::AnyRank);
        assert_eq!(ShapeSpec::parse("[]").unwrap(), ShapeSpec::AnyRank);
    }

    #[test]
    fn parses_wildcard_dims() {
        let spec = ShapeSpec::parse("[*,*,3]").unwrap();
        assert_eq!(spec.rank(), Some(3));
        assert_eq!(spec.to_string(), "[*, *, 3]");
        assert!(spec.matches_lengths(&[10, 20, 3]));
        assert!(!spec.matches_lengths(&[10, 20, 4]));
        assert!(!spec.matches_lengths(&[10, 3]));
    }

    #[test]
    fn rejects_rank_zero() {
        let err = ShapeSpec::parse("[rank=0]").unwrap_err();
        assert!(err.contains("Scalar"), "{}", err);
    }

    #[test]
    fn all_wildcard_ranks_display_short() {
        assert_eq!(ShapeSpec::from_rank(2).to_string(), "[rank=2]");
        assert_eq!(ShapeSpec::from_lengths(&[2, 3]).to_string(), "[2, 3]");
    }

    #[test]
    fn display_types() {
        assert_eq!(PType::Scalar.to_string(), "Scalar");
        assert_eq!(PType::Arg(ArgKind::Str).to_string(), "StrArg");
        assert_eq!(PType::Bytes.to_string(), "Bytes");
        assert_eq!(PType::Tensor(ShapeSpec::AnyRank).to_string(), "Tensor");
        assert_eq!(
            PType::Image(ShapeSpec::parse("[*,*,3]").unwrap()).to_string(),
            "Image[*, *, 3]"
        );
    }

    #[test]
    fn loop_var_lowers_rank_by_one() {
        let tensor3 = PType::Tensor(ShapeSpec::from_rank(3));
        assert_eq!(
            tensor3.each_loop_var().unwrap(),
            PType::Tensor(ShapeSpec::from_rank(2))
        );

        let tensor1 = PType::Tensor(ShapeSpec::from_rank(1));
        assert_eq!(tensor1.each_loop_var().unwrap(), PType::Scalar);

        let image3 = PType::Image(ShapeSpec::from_rank(3));
        assert_eq!(
            image3.each_loop_var().unwrap(),
            PType::Image(ShapeSpec::from_rank(2))
        );

        let audio2 = PType::Audio(ShapeSpec::from_rank(2));
        assert_eq!(
            audio2.each_loop_var().unwrap(),
            PType::Audio(ShapeSpec::from_rank(1))
        );

        let audio1 = PType::Audio(ShapeSpec::from_rank(1));
        assert_eq!(audio1.each_loop_var().unwrap(), PType::Scalar);
    }

    #[test]
    fn loop_var_rejects_unknown_and_flat_types() {
        assert!(PType::Tensor(ShapeSpec::AnyRank)
            .each_loop_var()
            .unwrap_err()
            .contains("not known"));
        assert!(PType::Bytes.each_loop_var().is_err());
        assert!(PType::Composite.each_loop_var().is_err());
        assert!(PType::Image(ShapeSpec::from_rank(1))
            .each_loop_var()
            .unwrap_err()
            .contains("rank >= 2"));
    }

    #[test]
    fn verifies_tensor_shapes() {
        use crate::Tensor;
        let tensor = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let spec = PType::Tensor(ShapeSpec::parse("[2,3]").unwrap());
        assert!(spec
            .verify_payload(&Payload::Tensor(tensor.clone()))
            .is_ok());

        let wrong = PType::Tensor(ShapeSpec::parse("[3,2]").unwrap());
        assert!(wrong.verify_payload(&Payload::Tensor(tensor)).is_err());
    }

    #[test]
    fn verifies_image_channels_in_both_layouts() {
        use crate::ColorSpace;
        // A 2x2 RGB image, HWC and CHW, holding the same 12 samples.
        let hwc = crate::Image::from_f32_hwc(
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
            2,
            2,
            ColorSpace::Rgb,
        )
        .unwrap();
        let chw = crate::Image::from_f32_chw(
            &[1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            2,
            2,
            ColorSpace::Rgb,
        )
        .unwrap();
        let spec = PType::Image(ShapeSpec::parse("[*,*,3]").unwrap());
        assert!(spec.verify_payload(&Payload::Image(hwc.clone())).is_ok());
        assert!(spec.verify_payload(&Payload::Image(chw)).is_ok());

        let spec = PType::Image(ShapeSpec::parse("[*,*,4]").unwrap());
        assert!(spec.verify_payload(&Payload::Image(hwc)).is_err());
    }

    #[test]
    fn verifies_scalar_rank() {
        use crate::Tensor;
        let scalar = Tensor::from_f32_shape(&[7.0], vec![]).unwrap();
        assert!(PType::Scalar
            .verify_payload(&Payload::Scalar(scalar.clone()))
            .is_ok());
        // A host may hand over an untagged rank-0 tensor.
        assert!(PType::Scalar
            .verify_payload(&Payload::Tensor(scalar))
            .is_ok());
        // A rank-1 tensor is not a scalar.
        assert!(PType::Scalar
            .verify_payload(&Payload::Tensor(Tensor::from_f32_slice(&[7.0])))
            .is_err());
        // A tagged scalar is not a Tensor.
        assert!(PType::Tensor(ShapeSpec::from_rank(1))
            .verify_payload(&Payload::Scalar(
                Tensor::from_f32_shape(&[7.0], vec![]).unwrap()
            ))
            .is_err());
    }

    #[test]
    fn verifies_arg_kinds() {
        let int = Payload::Arg(RVec::from(b"44100".to_vec()));
        let bool = Payload::Arg(RVec::from(b"true".to_vec()));
        assert!(PType::Arg(ArgKind::Int).verify_payload(&int).is_ok());
        assert!(PType::Arg(ArgKind::Float).verify_payload(&int).is_ok());
        assert!(PType::Arg(ArgKind::Str).verify_payload(&int).is_ok());
        assert!(PType::Arg(ArgKind::Bool).verify_payload(&int).is_err());
        assert!(PType::Arg(ArgKind::Bool).verify_payload(&bool).is_ok());
        assert!(PType::Tensor(ShapeSpec::AnyRank)
            .verify_payload(&int)
            .is_err());
    }

    #[test]
    fn refines_tensor_output_with_shape() {
        let refined = PType::from_data_type(DataType::Tensor | DataType::Scalar)
            .with_shape(&Shape::new(vec![2, 3]));
        assert_eq!(refined, PType::Tensor(ShapeSpec::from_lengths(&[2, 3])));

        let scalar =
            PType::from_data_type(DataType::Tensor | DataType::Scalar).with_shape(&Shape::scalar());
        assert_eq!(scalar, PType::Scalar);
    }

    #[test]
    fn shapes_define_element_count() {
        assert_eq!(Shape::scalar().element_count(), Some(1));
        assert_eq!(Shape::new(vec![2, 3, 4]).element_count(), Some(24));
        assert!(Shape::scalar().is_scalar());
        assert_eq!(Shape::new(vec![5]).to_string(), "[5]");
    }
}
