//! Engine-enforced shape analysis and invocation-local execution data.
use crate::{
    ActionArgs, DataType, Dimension, Payload, RString, RVec, ShapeResult, StableAbi, TensorDType,
    Tuple2, ValueShape, ValueShapeResult,
};
use abi_stable::std_types::ROption;
pub const ACTION_ABI_VERSION: u32 = 1;
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub enum Metadata {
    Unknown,
    Tensor {
        dtype: TensorDType,
    },
    Image {
        dtype: TensorDType,
        color_space: crate::ColorSpace,
        layout: crate::ImageLayout,
    },
    Audio {
        dtype: TensorDType,
        sample_rate: u32,
        layout: crate::AudioLayout,
        channel_layout: crate::AudioChannelLayout,
    },
    Composite(RVec<Metadata>),
}
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct InputDescriptor {
    pub value: ValueShape,
    pub metadata: Metadata,
}
impl Metadata {
    fn partial(value: &ValueShape) -> Self {
        match value {
            ValueShape::Composite(items) => {
                Self::Composite(items.iter().map(Self::partial).collect())
            }
            _ => Self::Unknown,
        }
    }
    pub fn get(&self, index: usize) -> Option<&Self> {
        match self {
            Self::Composite(items) => items.get(index),
            _ => None,
        }
    }
}
impl std::ops::Index<usize> for Metadata {
    type Output = Self;
    fn index(&self, index: usize) -> &Self {
        self.get(index)
            .expect("metadata component index out of bounds")
    }
}
impl InputDescriptor {
    pub fn partial(value: ValueShape) -> Self {
        Self {
            metadata: Metadata::partial(&value),
            value,
        }
    }
    pub fn from_payload(payload: &Payload) -> Self {
        let metadata = match payload.unwrap_payload() {
            Payload::Tensor(t) | Payload::Scalar(t) => Metadata::Tensor { dtype: t.dtype },
            Payload::Audio(a) => Metadata::Audio {
                dtype: a.tensor.dtype,
                sample_rate: a.sample_rate,
                layout: a.layout,
                channel_layout: a.channel_layout,
            },
            Payload::Image(i) => Metadata::Image {
                dtype: i.tensor.dtype,
                color_space: i.color_space,
                layout: i.layout,
            },
            Payload::Composite(items) => Metadata::Composite(
                items
                    .iter()
                    .map(|p| Self::from_payload(p).metadata)
                    .collect(),
            ),
            _ => Metadata::Unknown,
        };
        Self {
            value: ValueShape::from_payload(payload),
            metadata,
        }
    }
}
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub enum PreparedValue {
    Integer(i64),
    Unsigned(u64),
    Float(f64),
    Bool(bool),
    Text(RString),
    List(RVec<PreparedValue>),
    Fields(RVec<Tuple2<RString, PreparedValue>>),
}
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct PreparedData {
    pub output: ValueShape,
    pub args: PreparedArgs,
    pub fields: RVec<Tuple2<RString, PreparedValue>>,
}
impl PreparedData {
    pub fn field(&self, name: &str) -> Option<&PreparedValue> {
        self.fields
            .iter()
            .find_map(|Tuple2(key, value)| (key == name).then_some(value))
    }
    pub fn float(&self, name: &str) -> Option<f64> {
        match self.field(name) {
            Some(PreparedValue::Float(value)) => Some(*value),
            _ => None,
        }
    }
    pub fn unsigned(&self, name: &str) -> Option<u64> {
        match self.field(name) {
            Some(PreparedValue::Unsigned(value)) => Some(*value),
            _ => None,
        }
    }
    pub fn output_dims(&self) -> Result<Vec<usize>, RString> {
        self.output
            .shape()
            .and_then(|s| s.known_dims())
            .ok_or_else(|| "Execution plan requires concrete output dimensions".into())
    }
}
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub enum ShapeCheckResult {
    Invalid {
        reason: RString,
    },
    Deferred {
        output: ValueShape,
        unresolved: RVec<RString>,
    },
    Ready {
        output: ValueShape,
        prepared: PreparedData,
    },
}
impl ShapeCheckResult {
    pub fn prediction(&self) -> ValueShapeResult {
        match self {
            Self::Invalid { reason } => ValueShapeResult::Invalid(reason.clone()),
            Self::Deferred { output, .. } | Self::Ready { output, .. } => {
                ValueShapeResult::Ok(output.clone())
            }
        }
    }
}
pub type ShapeCheckFn = extern "C" fn(InputDescriptor, ActionArgs) -> ShapeCheckResult;
/// Layout checked before either action callback can be invoked.
#[repr(C)]
#[derive(StableAbi)]
pub struct ActionAbiLayout {
    pub input: InputDescriptor,
    pub arguments: ActionArgs,
    pub result: ShapeCheckResult,
    pub payload: Payload,
    pub prepared: PreparedData,
}
fn partial(value: &ValueShape) -> bool {
    match value {
        ValueShape::Unknown => true,
        ValueShape::Composite(items) => items.iter().any(partial),
        ValueShape::Leaf { kind, shape } => match shape {
            ROption::RSome(s) => s.dims().contains(&Dimension::Unknown),
            ROption::RNone => *kind != DataType::Bytes,
        },
    }
}
/// Shared bridge for dimensional analyses. The engine always calls the exported
/// shapecheck; existing pure analyses remain private action implementation details.
pub fn analyze(
    input: InputDescriptor,
    args: ActionArgs,
    name: &str,
    accepted: DataType,
    output_kind: DataType,
    single: Option<NativeShapeFn>,
    tree: Option<NativeValueFn>,
) -> ShapeCheckResult {
    let kind = match &input.value {
        ValueShape::Unknown => None,
        ValueShape::Composite(_) => Some(DataType::Composite),
        ValueShape::Leaf { kind, .. } => Some(*kind),
    };
    if kind.is_some_and(|k| !accepted.intersects(k)) {
        return ShapeCheckResult::Invalid {
            reason: format!("Action '{name}' does not accept {}", kind.unwrap()).into(),
        };
    }
    let args = PreparedArgs::from(args);
    let result = if let Some(get) = tree {
        get(input.value.clone(), args.clone())
    } else if let (Some(get), Some(shape)) = (single, input.value.shape()) {
        match get(shape.clone(), args.clone()) {
            ShapeResult::Invalid(reason) => ValueShapeResult::Invalid(reason),
            ShapeResult::Unknown => ValueShapeResult::Ok(ValueShape::unranked(output_kind)),
            ShapeResult::Ok(shape) => {
                let kind = if output_kind == (DataType::Tensor | DataType::Scalar) {
                    if shape.rank() == 0 {
                        DataType::Scalar
                    } else {
                        DataType::Tensor
                    }
                } else {
                    output_kind
                };
                ValueShapeResult::Ok(ValueShape::Leaf {
                    kind,
                    shape: Some(shape).into(),
                })
            }
        }
    } else {
        ValueShapeResult::Ok(ValueShape::unranked(output_kind))
    };
    let output = match result {
        ValueShapeResult::Invalid(reason) => return ShapeCheckResult::Invalid { reason },
        ValueShapeResult::Ok(output) => output,
        ValueShapeResult::Unknown => ValueShape::unranked(output_kind),
    };
    let dynamic_args = args
        .positional
        .iter()
        .chain(args.named.iter().map(|Tuple2(_, v)| v))
        .any(|s| s.starts_with('$'));
    if partial(&input.value) || dynamic_args {
        return ShapeCheckResult::Deferred {
            output,
            unresolved: vec![
                "Concrete input dimensions, metadata or argument values are required".into(),
            ]
            .into(),
        };
    }
    let prepared = PreparedData {
        output: output.clone(),
        args,
        fields: RVec::new(),
    };
    ShapeCheckResult::Ready { output, prepared }
}

#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct CachedArgument {
    text: RString,
    signed: ROption<i64>,
    unsigned: ROption<u64>,
    float: ROption<f32>,
    list: ROption<RVec<u64>>,
}
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct PreparedArgs {
    pub raw: ActionArgs,
    cache: RVec<CachedArgument>,
}
impl std::ops::Deref for PreparedArgs {
    type Target = ActionArgs;
    fn deref(&self) -> &ActionArgs {
        &self.raw
    }
}
impl From<ActionArgs> for PreparedArgs {
    fn from(raw: ActionArgs) -> Self {
        let cache = raw
            .positional
            .iter()
            .chain(raw.named.iter().map(|Tuple2(_, v)| v))
            .map(|text| {
                let clean = text
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim_start_matches('(')
                    .trim_end_matches(')');
                let list = if clean.is_empty() {
                    Some(RVec::new())
                } else {
                    clean
                        .split(',')
                        .map(|s| s.trim().parse::<u64>())
                        .collect::<Result<RVec<_>, _>>()
                        .ok()
                };
                CachedArgument {
                    text: text.clone(),
                    signed: text.parse().ok().into(),
                    unsigned: text.parse().ok().into(),
                    float: text.parse().ok().into(),
                    list: list.into(),
                }
            })
            .collect();
        Self { raw, cache }
    }
}
pub trait CachedParse: std::str::FromStr + Sized {
    fn cached(arg: &CachedArgument) -> Option<Self>;
}
macro_rules! integer_cache {
    ($ty:ty,$field:ident) => {
        impl CachedParse for $ty {
            fn cached(arg: &CachedArgument) -> Option<Self> {
                arg.$field
                    .as_ref()
                    .into_option()
                    .and_then(|v| (*v).try_into().ok())
            }
        }
    };
}
integer_cache!(usize, unsigned);
integer_cache!(u32, unsigned);
integer_cache!(isize, signed);
impl CachedParse for f32 {
    fn cached(arg: &CachedArgument) -> Option<Self> {
        arg.float.as_ref().into_option().copied()
    }
}
pub trait ArgumentSource {
    fn raw(&self) -> &ActionArgs;
    fn parse<T: CachedParse>(&self, text: &str) -> Result<T, RString>;
}
impl ArgumentSource for ActionArgs {
    fn raw(&self) -> &ActionArgs {
        self
    }
    fn parse<T: CachedParse>(&self, text: &str) -> Result<T, RString> {
        text.parse()
            .map_err(|_| format!("Invalid numeric argument '{text}'").into())
    }
}
impl ArgumentSource for PreparedArgs {
    fn raw(&self) -> &ActionArgs {
        &self.raw
    }
    fn parse<T: CachedParse>(&self, text: &str) -> Result<T, RString> {
        self.cache
            .iter()
            .find(|arg| arg.text == text)
            .and_then(T::cached)
            .ok_or_else(|| format!("Invalid numeric argument '{text}'").into())
    }
}
impl<T: ArgumentSource> ArgumentSource for &T {
    fn raw(&self) -> &ActionArgs {
        (*self).raw()
    }
    fn parse<U: CachedParse>(&self, text: &str) -> Result<U, RString> {
        (*self).parse(text)
    }
}
impl PreparedArgs {
    pub fn parse<T: CachedParse>(&self, text: &str) -> Result<T, RString> {
        ArgumentSource::parse(self, text)
    }
    pub fn usize_list(&self, text: &str) -> Result<Vec<usize>, RString> {
        self.cache
            .iter()
            .find(|arg| arg.text == text)
            .and_then(|arg| arg.list.as_ref().into_option())
            .ok_or_else(|| RString::from("Invalid dimension list"))?
            .iter()
            .map(|n| {
                (*n).try_into()
                    .map_err(|_| RString::from("Dimension overflows usize"))
            })
            .collect()
    }
}
pub type NativeShapeFn = fn(crate::Shape, PreparedArgs) -> ShapeResult;
pub type NativeValueFn = fn(ValueShape, PreparedArgs) -> ValueShapeResult;

/// Engine dispatch: preparation is mandatory, local to this invocation, and
/// consumed by execution only after the callback returns Ready.
pub fn execute(
    name: &str,
    check: ShapeCheckFn,
    process: crate::ProcessFn,
    payload: Payload,
) -> Payload {
    match check(
        InputDescriptor::from_payload(&payload),
        payload.args().cloned().unwrap_or_default(),
    ) {
        ShapeCheckResult::Invalid { reason } => Payload::Error(reason),
        ShapeCheckResult::Deferred { unresolved, .. } => Payload::Error(
            format!("Action '{name}' could not prepare execution: {unresolved:?}").into(),
        ),
        ShapeCheckResult::Ready { output, prepared } => {
            let actual = process(payload.into_unwrapped(), prepared);
            if matches!(actual, Payload::Error(_)) {
                return actual;
            }
            match output.verify(&actual) {
                Ok(()) => actual,
                Err(reason) => Payload::Error(
                    format!("Action '{name}' shape contract mismatch: {reason}").into(),
                ),
            }
        }
    }
}

/// Add the normalized axis to a successful dimensional reduction plan. The
/// dimensional callback has already established axis compatibility and output
/// dimensions; execution consumes this index and the predicted dimensions.
pub fn axis_plan(
    result: ShapeCheckResult,
    rank: Option<usize>,
    position: usize,
    default: Option<isize>,
    clamp_negative: bool,
) -> ShapeCheckResult {
    match result {
        ShapeCheckResult::Ready {
            output,
            mut prepared,
        } => {
            let rank = rank.expect("ready reduction has known rank");
            let raw = prepared
                .args
                .get_named("axis")
                .or_else(|| prepared.args.get_named("dim"))
                .or_else(|| prepared.args.positional.get(position).map(|v| v.as_str()))
                .map(|v| {
                    prepared
                        .args
                        .parse::<isize>(v)
                        .expect("dimensional callback validated axis")
                })
                .or(default);
            if let Some(raw) = raw.filter(|_| rank > 0) {
                let axis = if raw < 0 {
                    let axis = raw.saturating_add(rank as isize);
                    if clamp_negative {
                        axis.max(0)
                    } else {
                        axis
                    }
                } else {
                    raw
                };
                prepared
                    .fields
                    .push(Tuple2("axis".into(), PreparedValue::Unsigned(axis as u64)));
            }
            ShapeCheckResult::Ready { output, prepared }
        }
        other => other,
    }
}

/// Normalize pooling parameters after dimensional compatibility is established.
pub fn pool2d_plan(result: ShapeCheckResult) -> ShapeCheckResult {
    match result {
        ShapeCheckResult::Ready {
            output,
            mut prepared,
        } => {
            for (name, aliases, position) in [
                ("kernel", &["kernel_size", "kernel"][..], 0),
                ("stride", &["stride"][..], 1),
            ] {
                let value = aliases
                    .iter()
                    .find_map(|name| prepared.args.get_named(name))
                    .or_else(|| {
                        prepared
                            .args
                            .positional
                            .get(position)
                            .map(|value| value.as_str())
                    })
                    .map(|value| {
                        prepared
                            .args
                            .parse::<usize>(value)
                            .expect("dimensional callback validated pooling parameter")
                    })
                    .unwrap_or(2)
                    .max(1);
                prepared
                    .fields
                    .push(Tuple2(name.into(), PreparedValue::Unsigned(value as u64)));
            }
            ShapeCheckResult::Ready { output, prepared }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern "C" fn must_not_execute(_: Payload, _: PreparedData) -> Payload {
        panic!("process called after rejected shapecheck")
    }
    extern "C" fn invalid(_: InputDescriptor, _: ActionArgs) -> ShapeCheckResult {
        ShapeCheckResult::Invalid {
            reason: "incompatible dimensions".into(),
        }
    }
    extern "C" fn deferred(input: InputDescriptor, _: ActionArgs) -> ShapeCheckResult {
        ShapeCheckResult::Deferred {
            output: input.value,
            unresolved: vec!["missing metadata".into()].into(),
        }
    }
    extern "C" fn ready(input: InputDescriptor, args: ActionArgs) -> ShapeCheckResult {
        let output = input.value;
        let token = args
            .get_named("n")
            .map(|n| n.parse::<u64>().unwrap())
            .unwrap_or(0);
        ShapeCheckResult::Ready {
            output: output.clone(),
            prepared: PreparedData {
                output,
                args: args.into(),
                fields: vec![Tuple2("token".into(), PreparedValue::Unsigned(token))].into(),
            },
        }
    }
    extern "C" fn consume(input: Payload, prepared: PreparedData) -> Payload {
        assert!(input.args().is_none());
        let n = prepared
            .args
            .parse::<usize>(prepared.args.get_named("n").unwrap())
            .unwrap();
        assert_eq!(prepared.unsigned("token"), Some(n as u64));
        let Payload::Scalar(value) = &input else {
            panic!("expected scalar")
        };
        assert_eq!(value.as_f32_slice().unwrap()[0], n as f32);
        input
    }
    extern "C" fn wrong_output(_: Payload, _: PreparedData) -> Payload {
        Payload::Data {
            buffer: vec![1, 2].into(),
        }
    }
    #[test]
    fn rejected_checks_never_execute() {
        for check in [invalid as ShapeCheckFn, deferred as ShapeCheckFn] {
            assert!(matches!(
                execute("test", check, must_not_execute, Payload::scalar_f32(1.0)),
                Payload::Error(_)
            ));
        }
    }
    #[test]
    fn prepared_data_is_consumed_per_invocation() {
        std::thread::scope(|scope| {
            for n in 1..=8 {
                scope.spawn(move || {
                    let input = Payload::WithArgs {
                        payload: crate::RBox::new(Payload::scalar_f32(n as f32)),
                        args: ActionArgs {
                            positional: RVec::new(),
                            named: vec![Tuple2("n".into(), n.to_string().into())].into(),
                        },
                    };
                    assert!(matches!(
                        execute("test", ready, consume, input),
                        Payload::Scalar(_)
                    ));
                });
            }
        });
    }
    extern "C" fn wrong_dimensions(_: Payload, _: PreparedData) -> Payload {
        Payload::Tensor(crate::Tensor::from_f32_slice(&[1.0, 2.0, 3.0]))
    }
    #[test]
    fn output_contract_is_verified() {
        let input = Payload::Tensor(crate::Tensor::from_f32_slice(&[1.0, 2.0]));
        assert!(matches!(
            execute("test", ready, wrong_dimensions, input),
            Payload::Error(_)
        ));
        assert!(matches!(
            execute("test", ready, wrong_output, Payload::scalar_f32(1.0)),
            Payload::Error(_)
        ));
        let shape = ValueShape::Composite(
            vec![
                ValueShape::Leaf {
                    kind: DataType::Scalar,
                    shape: Some(crate::Shape::new(Vec::<usize>::new())).into(),
                },
                ValueShape::tensor(crate::Shape::new([2usize])),
            ]
            .into(),
        );
        let wrong = Payload::Composite(
            vec![
                Payload::Tensor(crate::Tensor::from_f32_shape(&[1.0, 2.0], vec![2]).unwrap()),
                Payload::scalar_f32(1.0),
            ]
            .into(),
        );
        assert!(shape.verify(&wrong).is_err());
    }
    #[test]
    fn descriptors_keep_ordered_typed_metadata_without_numeric_contents() {
        let input = Payload::Composite(
            vec![
                Payload::Tensor(crate::Tensor::from_f32_slice(&[1.0, 2.0])),
                Payload::Audio(crate::Audio::from_f32_planar(&[1.0; 3], 1, 48000).unwrap()),
            ]
            .into(),
        );
        let descriptor = InputDescriptor::from_payload(&input);
        assert!(matches!(
            descriptor.metadata[0],
            Metadata::Tensor {
                dtype: TensorDType::F32
            }
        ));
        assert!(matches!(
            descriptor.metadata[1],
            Metadata::Audio {
                sample_rate: 48000,
                layout: crate::AudioLayout::Planar,
                ..
            }
        ));
        assert_eq!(
            descriptor.value[1].shape().unwrap().dims(),
            &[Dimension::Known(3)]
        );
        let partial = InputDescriptor::partial(descriptor.value);
        assert!(matches!(partial.metadata[0], Metadata::Unknown));
        assert!(matches!(partial.metadata[1], Metadata::Unknown));
    }
    #[test]
    fn cached_numbers_preserve_float_and_list_semantics() {
        let args: PreparedArgs = ActionArgs {
            positional: vec![
                "1.23456789".into(),
                "[2, 0, 1]".into(),
                "-1".into(),
                "18446744073709551616".into(),
            ]
            .into(),
            named: RVec::new(),
        }
        .into();
        assert_eq!(
            args.parse::<f32>("1.23456789").unwrap().to_bits(),
            "1.23456789".parse::<f32>().unwrap().to_bits()
        );
        assert_eq!(args.usize_list("[2, 0, 1]").unwrap(), vec![2, 0, 1]);
        assert!(args.parse::<usize>("-1").is_err());
        assert!(args.parse::<usize>("18446744073709551616").is_err());
    }
}
