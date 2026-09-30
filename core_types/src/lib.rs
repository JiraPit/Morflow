pub mod audio;
pub mod image;
pub mod shape;
pub mod tensor;

pub use abi_stable;
pub use abi_stable::std_types::{RBox, RString, RVec, Tuple2};
pub use abi_stable::StableAbi;

pub use audio::{Audio, AudioChannelLayout, AudioLayout};
pub use image::{ColorSpace, Image, ImageLayout};
pub use shape::{
    arg_text, reducer_axis_reason, scalar_number, scalar_text, tensor_scalar_text, ArgKind, Dim,
    GetShapeFn, PType, Shape, ShapeResult, ShapeSpec,
};
pub use tensor::{compute_c_contiguous_strides, parse_shape_str, Tensor, TensorDType};

#[repr(transparent)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DataType(pub u8);

#[allow(non_upper_case_globals)]
impl DataType {
    pub const Bytes: DataType = DataType(1 << 0);
    pub const Tensor: DataType = DataType(1 << 1);
    pub const Composite: DataType = DataType(1 << 2);
    pub const Image: DataType = DataType(1 << 3);
    pub const Audio: DataType = DataType(1 << 4);
    /// A rank-0 numeric value, i.e. a single number rather than a tensor.
    pub const Scalar: DataType = DataType(1 << 5);
    pub const Any: DataType = DataType(0b00111111);

    #[inline]
    pub const fn contains(&self, other: DataType) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn intersects(&self, other: DataType) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub const fn bits(&self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
}

impl std::ops::BitOr for DataType {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        DataType(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for DataType {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for DataType {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        DataType(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for DataType {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == Self::Any.0 {
            return write!(f, "Any");
        }
        let mut parts = Vec::new();
        if self.contains(Self::Bytes) {
            parts.push("Bytes");
        }
        if self.contains(Self::Tensor) {
            parts.push("Tensor");
        }
        if self.contains(Self::Composite) {
            parts.push("Composite");
        }
        if self.contains(Self::Image) {
            parts.push("Image");
        }
        if self.contains(Self::Audio) {
            parts.push("Audio");
        }
        if self.contains(Self::Scalar) {
            parts.push("Scalar");
        }
        if parts.is_empty() {
            write!(f, "Unknown")
        } else {
            write!(f, "{}", parts.join(" | "))
        }
    }
}

/// Positional and named arguments passed to an action invocation.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, Default)]
pub struct ActionArgs {
    pub positional: RVec<RString>,
    pub named: RVec<Tuple2<RString, RString>>,
}

impl ActionArgs {
    pub fn get_named(&self, key: &str) -> Option<&str> {
        for Tuple2(k, v) in &self.named {
            if k.as_str() == key {
                return Some(v.as_str());
            }
        }
        None
    }
}

/// Universal payload enum passed between the engine host and dynamic actions across FFI.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub enum Payload {
    Data {
        buffer: RVec<u8>,
    },
    Tensor(Tensor),
    Image(Image),
    Audio(Audio),
    Composite(RVec<Payload>),
    WithArgs {
        payload: RBox<Payload>,
        args: ActionArgs,
    },
    Error(RString),
    /// A single pipeline argument, held as its textual form. Arguments are only
    /// readable by actions through their named/positional arguments; they never
    /// flow through a chain.
    Arg(RVec<u8>),
    /// A rank-0 numeric value.
    Scalar(Tensor),
}

impl Payload {
    /// Wraps a rank-0 tensor as a scalar payload.
    pub fn scalar(tensor: Tensor) -> Result<Self, RString> {
        if tensor.rank() != 0 {
            return Err(RString::from(format!(
                "A scalar value must be rank 0, but this tensor has rank {}",
                tensor.rank()
            )));
        }
        Ok(Payload::Scalar(tensor))
    }

    /// Wraps a single f32 as a scalar payload.
    pub fn scalar_f32(value: f32) -> Self {
        Payload::Scalar(Tensor::from_f32_shape(&[value], vec![]).expect("rank-0 shape is valid"))
    }

    /// Wraps textual argument bytes.
    pub fn arg(text: impl AsRef<str>) -> Self {
        Payload::Arg(RVec::from(text.as_ref()))
    }

    /// The underlying tensor, whether it is tagged as a tensor or a scalar.
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self.unwrap_payload() {
            Payload::Tensor(t) | Payload::Scalar(t) => Some(t),
            _ => None,
        }
    }

    /// The underlying tensor for in-place mutation, whether tagged as a tensor
    /// or a scalar.
    pub fn as_tensor_mut(&mut self) -> Option<&mut Tensor> {
        match self {
            Payload::Tensor(t) | Payload::Scalar(t) => Some(t),
            Payload::WithArgs { payload, .. } => payload.as_tensor_mut(),
            _ => None,
        }
    }

    /// The underlying tensor only if this is a scalar.
    pub fn as_scalar(&self) -> Option<&Tensor> {
        match self.unwrap_payload() {
            Payload::Scalar(t) => Some(t),
            _ => None,
        }
    }

    /// The underlying bytes only if this is an argument.
    pub fn as_arg(&self) -> Option<&[u8]> {
        match self.unwrap_payload() {
            Payload::Arg(bytes) => Some(bytes.as_slice()),
            _ => None,
        }
    }

    /// Tags a tensor with the payload variant matching its rank: a rank-0
    /// tensor is a `Scalar` value, anything else is a `Tensor`.
    pub fn from_tensor(tensor: Tensor) -> Self {
        if tensor.rank() == 0 {
            Payload::Scalar(tensor)
        } else {
            Payload::Tensor(tensor)
        }
    }

    /// Extracts the core underlying payload if wrapped with arguments.
    pub fn unwrap_payload(&self) -> &Payload {
        match self {
            Payload::WithArgs { payload, .. } => payload.unwrap_payload(),
            other => other,
        }
    }

    /// Consumes and extracts the core underlying payload if wrapped with arguments without cloning.
    pub fn into_unwrapped(self) -> Payload {
        match self {
            Payload::WithArgs { payload, .. } => RBox::into_inner(payload).into_unwrapped(),
            other => other,
        }
    }

    /// Deconstructs the payload into its underlying payload and optional action arguments without cloning.
    pub fn take_payload_and_args(self) -> (Payload, Option<ActionArgs>) {
        match self {
            Payload::WithArgs { payload, args } => {
                let inner = RBox::into_inner(payload).into_unwrapped();
                (inner, Some(args))
            }
            other => (other, None),
        }
    }

    /// Access optional action arguments if present.
    pub fn args(&self) -> Option<&ActionArgs> {
        match self {
            Payload::WithArgs { args, .. } => Some(args),
            _ => None,
        }
    }

    /// Resolves how an `each` loop iterates this payload, without discarding its
    /// type. Returns the axis to slice and the number of iterations.
    ///
    /// The axis is the *channel* axis for images and for multi-channel audio, so
    /// that every iteration is itself a well-formed payload of the same type.
    /// Single-channel audio is the exception: mono is a rank-1 tensor, so there
    /// is no channel axis and axis 0 is the sample axis.
    pub fn each_axis(&self) -> Result<(usize, usize), RString> {
        match self.unwrap_payload() {
            Payload::Tensor(t) => {
                if t.rank() == 0 {
                    return Err(RString::from(
                        "Cannot loop over a scalar value; 'each' requires a rank >= 1 payload",
                    ));
                }
                Ok((0, t.shape[0]))
            }
            Payload::Scalar(_) => Err(RString::from(
                "Cannot loop over a scalar value; 'each' requires a rank >= 1 payload",
            )),
            Payload::Arg(_) => Err(RString::from(
                "Cannot loop over an argument; arguments are plain values, not payloads",
            )),
            Payload::Audio(a) => {
                if a.tensor.rank() == 0 {
                    return Err(RString::from(
                        "Cannot loop over a scalar/empty audio payload",
                    ));
                }
                // Rank-1 audio is mono: axis 0 is the sample axis. Rank-2 audio
                // has a layout-dependent channel axis.
                let axis = match (a.tensor.rank(), a.layout) {
                    (1, _) => 0,
                    (_, AudioLayout::Planar) => 0,
                    (_, AudioLayout::Interleaved) => 1,
                };
                Ok((axis, a.tensor.shape[axis]))
            }
            Payload::Image(i) => {
                if i.tensor.rank() < 2 {
                    return Err(RString::from(
                        "Cannot loop over a rank-1 image payload; 'each' requires rank >= 2",
                    ));
                }
                // Iterate channels so each slice is a valid rank-2 grayscale image.
                let axis = match i.layout {
                    ImageLayout::Hwc => 2,
                    ImageLayout::Chw => 0,
                };
                Ok((axis, i.tensor.shape[axis]))
            }
            Payload::WithArgs { .. } => unreachable!("unwrap_payload removes WithArgs"),
            Payload::Data { .. } => Err(RString::from(
                "Cannot execute 'each' on a Bytes payload; decode it into Tensor, Image, or Audio first",
            )),
            Payload::Composite(_) => Err(RString::from(
                "Cannot execute 'each' on a composite payload",
            )),
            Payload::Error(e) => Err(RString::from(format!(
                "Cannot execute 'each' on an error payload: {}",
                e
            ))),
        }
    }

    /// Produces iteration `index` of an `each` loop as a payload of the same
    /// type where the slice shape permits it.
    ///
    /// A rank-1 audio slice becomes a mono `Audio` and a rank-2 image slice
    /// becomes a grayscale `Image`. Slices that have no dimensions left — a
    /// rank-1 `Tensor` or mono `Audio` iterated to single values — become
    /// `Scalar`, which is the matching value of those types.
    pub fn each_slice(&self, index: usize) -> Result<Payload, RString> {
        let (axis, _) = self.each_axis()?;
        match self.unwrap_payload() {
            Payload::Tensor(t) => {
                let sliced = t.slice_axis_index(axis, index)?;
                Ok(Payload::from_tensor(sliced))
            }
            Payload::Audio(a) => {
                let sliced = a.tensor.slice_axis_index(axis, index)?;
                if sliced.rank() == 0 {
                    return Ok(Payload::Scalar(sliced));
                }
                let channel_layout = AudioChannelLayout::from_channel_count(1);
                let mono =
                    Audio::new(sliced, a.sample_rate, channel_layout, a.layout).map_err(|e| {
                        RString::from(format!("Cannot iterate this audio payload: {}", e))
                    })?;
                Ok(Payload::Audio(mono))
            }
            Payload::Image(i) => {
                let sliced = i.tensor.slice_axis_index(axis, index)?;
                let channel = Image::new(sliced.clone(), ColorSpace::Grayscale, ImageLayout::Hwc)
                    .map_err(|e| {
                    RString::from(format!("Cannot iterate this image payload: {}", e))
                })?;
                Ok(Payload::Image(channel))
            }
            _ => Err(RString::from(
                "Cannot execute 'each' on a non-Tensor/non-Image/non-Audio payload",
            )),
        }
    }

    /// Restacks the outputs of an `each` loop back into a single payload of the
    /// *loop input's* type.
    ///
    /// The loop input alone decides the result: every iteration must return
    /// that same kind, so a loop over a `Tensor` always produces a `Tensor` and
    /// a loop over an `Image` always produces an `Image`. Iterating a rank-1
    /// payload produces `Scalar` slices, which are accepted here and restacked
    /// into the original rank-1 payload.
    pub fn each_restack(&self, items: &[Payload]) -> Result<Payload, RString> {
        if items.is_empty() {
            return Err(RString::from(
                "Cannot restack an 'each' loop that produced no iterations",
            ));
        }
        let parent = self.unwrap_payload();

        // Collect every iteration's tensor, rejecting any mismatched variant.
        let collect = |want: fn(&Payload) -> Option<&Tensor>,
                       parent_kind: &str|
         -> Result<Vec<Tensor>, RString> {
            let mut tensors = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                match want(item) {
                    Some(t) => tensors.push(t.clone()),
                    None => {
                        return Err(RString::from(format!(
                            "Cannot restack 'each' loop: iteration {} returned {} but the loop input is {}, so every iteration must return {}",
                            i,
                            payload_kind_name(item),
                            parent_kind,
                            parent_kind
                        )))
                    }
                }
            }
            Ok(tensors)
        };

        match parent {
            // A Tensor loop restacks on axis 0 into a Tensor. Iterations over a
            // rank-1 tensor are rank-0 scalars, which stack back to rank 1.
            Payload::Tensor(parent_tensor) => {
                let rank = parent_tensor.rank();
                let tensors = collect(tensor_of, "Tensor")?;
                for (i, tensor) in tensors.iter().enumerate() {
                    let expected = if rank == 1 { 0 } else { rank - 1 };
                    if tensor.rank() != expected {
                        return Err(RString::from(format!(
                            "Cannot restack 'each' loop: iteration {} returned a rank-{} value, but iterating a rank-{} Tensor produces rank-{} slices",
                            i,
                            tensor.rank(),
                            rank,
                            expected
                        )));
                    }
                }
                Ok(Payload::Tensor(Tensor::stack(&tensors, 0)?))
            }
            // Mono audio is rank 1, so its per-sample slices are rank-0 scalars.
            // Rank-2 audio iterates channels, so every slice is a mono Audio.
            Payload::Audio(parent_audio) => {
                if parent_audio.tensor.rank() == 1 {
                    let tensors = collect(tensor_of, "Audio")?;
                    for (i, tensor) in tensors.iter().enumerate() {
                        if tensor.rank() != 0 {
                            return Err(RString::from(format!(
                                "Cannot restack 'each' loop: iteration {} returned a rank-{} value, but iterating mono Audio produces single samples",
                                i,
                                tensor.rank()
                            )));
                        }
                    }
                    let stacked = Tensor::stack(&tensors, 0)?;
                    let audio = Audio::new(
                        stacked,
                        parent_audio.sample_rate,
                        AudioChannelLayout::Mono,
                        parent_audio.layout,
                    )?;
                    return Ok(Payload::Audio(audio));
                }

                let first_audio = match &items[0] {
                    Payload::Audio(a) => a,
                    other => {
                        return Err(RString::from(format!(
                            "Cannot restack 'each' loop: iteration 0 returned {} but the loop input is Audio, so every iteration must return Audio",
                            payload_kind_name(other)
                        )))
                    }
                };
                for (i, item) in items.iter().enumerate() {
                    let a = match item {
                        Payload::Audio(a) => a,
                        other => {
                            return Err(RString::from(format!(
                                "Cannot restack 'each' loop: iteration {} returned {} but the loop input is Audio, so every iteration must return Audio",
                                i,
                                payload_kind_name(other)
                            )))
                        }
                    };
                    if a.sample_rate != first_audio.sample_rate {
                        return Err(RString::from(format!(
                            "Cannot restack 'each' loop: iteration {} has sample rate {} but iteration 0 has {}",
                            i, a.sample_rate, first_audio.sample_rate
                        )));
                    }
                    if a.layout != first_audio.layout {
                        return Err(RString::from(format!(
                            "Cannot restack 'each' loop: iteration {} has audio layout {:?} but iteration 0 has {:?}",
                            i, a.layout, first_audio.layout
                        )));
                    }
                }

                // Every iteration is one channel, so stacking on axis 0 yields a
                // planar [channels, samples] buffer. Interleaved audio has to be
                // transposed back into [samples, channels].
                let stacked = Tensor::stack(&collect(audio_tensor_of, "Audio")?, 0)?;
                let (buffer, channel_layout) = match first_audio.layout {
                    AudioLayout::Planar => {
                        // A rank-1 result is mono regardless of how many
                        // iterations produced it.
                        let channel_layout = if stacked.rank() == 1 {
                            AudioChannelLayout::Mono
                        } else {
                            let channels = stacked.shape.first().copied().unwrap_or(1);
                            AudioChannelLayout::from_channel_count(channels)
                        };
                        (stacked, channel_layout)
                    }
                    AudioLayout::Interleaved => {
                        let interleaved = stacked.transpose(0, 1)?;
                        let channels = interleaved.shape.get(1).copied().unwrap_or(1);
                        (
                            interleaved,
                            AudioChannelLayout::from_channel_count(channels),
                        )
                    }
                };
                let audio = Audio::new(
                    buffer,
                    first_audio.sample_rate,
                    channel_layout,
                    first_audio.layout,
                )?;
                Ok(Payload::Audio(audio))
            }
            // An image loop iterates channels, so every slice is a grayscale
            // Image and the result rejoins on the parent's channel axis.
            Payload::Image(parent_image) => {
                let tensors = collect(image_tensor_of, "Image")?;
                let axis = match parent_image.layout {
                    ImageLayout::Hwc => 2,
                    ImageLayout::Chw => 0,
                };
                let stacked = Tensor::stack(&tensors, axis)?;
                let image = Image::new(stacked, parent_image.color_space, parent_image.layout)?;
                Ok(Payload::Image(image))
            }
            other => Err(RString::from(format!(
                "Cannot restack 'each' loop over {}: the loop input has no iteration axis",
                payload_kind_name(other)
            ))),
        }
    }
}

/// Human-readable payload variant name for error messages.
pub fn payload_kind_name(payload: &Payload) -> &'static str {
    match payload {
        Payload::Data { .. } => "Data",
        Payload::Tensor(_) => "Tensor",
        Payload::Image(_) => "Image",
        Payload::Audio(_) => "Audio",
        Payload::Composite(_) => "Composite",
        Payload::WithArgs { .. } => "WithArgs",
        Payload::Error(_) => "Error",
        Payload::Arg(_) => "Arg",
        Payload::Scalar(_) => "Scalar",
    }
}

/// The tensor of a `Tensor` or `Scalar` payload, which are the two variants that
/// carry a tensor.
fn tensor_of(payload: &Payload) -> Option<&Tensor> {
    match payload {
        Payload::Tensor(t) | Payload::Scalar(t) => Some(t),
        _ => None,
    }
}

fn audio_tensor_of(payload: &Payload) -> Option<&Tensor> {
    match payload {
        Payload::Audio(a) => Some(&a.tensor),
        _ => None,
    }
}

fn image_tensor_of(payload: &Payload) -> Option<&Tensor> {
    match payload {
        Payload::Image(i) => Some(&i.tensor),
        _ => None,
    }
}

pub type ProcessFn = extern "C" fn(Payload) -> Payload;
pub type GetTypeFn = extern "C" fn() -> DataType;

#[cfg(test)]
mod tests {
    use super::*;

    /// The payload enum crosses the FFI boundary as a by-value `repr(C)` type, so
    /// its size has to stay pinned; adding a variant wider than the current
    /// largest one would change it.
    #[test]
    fn payload_size_is_stable() {
        assert_eq!(std::mem::size_of::<Payload>(), 112);
    }

    #[test]
    fn scalar_accessor_reads_both_tensor_tags() {
        let mut tensor = Payload::Tensor(Tensor::from_f32_slice(&[1.0, 2.0]));
        let scalar = Payload::scalar_f32(3.0);
        assert!(tensor.as_tensor().is_some());
        assert!(scalar.as_tensor().is_some());
        assert!(tensor.as_scalar().is_none());
        assert!(scalar.as_scalar().is_some());
        assert_eq!(tensor.as_tensor().unwrap().rank(), 1);
        assert_eq!(scalar.as_tensor().unwrap().rank(), 0);

        tensor.as_tensor_mut().unwrap().as_f32_slice_mut()[0] = 42.0;
        assert_eq!(
            tensor.as_tensor().unwrap().as_f32_slice().unwrap(),
            &[42.0, 2.0]
        );
        assert!(scalar.as_arg().is_none());
    }

    #[test]
    fn scalar_requires_rank_zero() {
        assert!(Payload::scalar(Tensor::from_f32_slice(&[1.0])).is_err());
        let ok = Tensor::from_f32_shape(&[1.0], vec![]).unwrap();
        assert!(Payload::scalar(ok).is_ok());
    }

    #[test]
    fn tensor_of_reads_through_with_args() {
        let wrapped = Payload::WithArgs {
            payload: RBox::new(Payload::Scalar(
                Tensor::from_f32_shape(&[9.0], vec![]).unwrap(),
            )),
            args: ActionArgs::default(),
        };
        assert_eq!(wrapped.as_tensor().unwrap().rank(), 0);
        assert!(wrapped.as_arg().is_none());
    }

    #[test]
    fn rank_one_tensor_slices_are_scalars() {
        let parent = Payload::Tensor(Tensor::from_f32_shape(&[1.0, 2.0, 3.0], vec![3]).unwrap());
        assert_eq!(parent.each_axis().unwrap(), (0, 3));
        let first = parent.each_slice(0).unwrap();
        assert!(matches!(first, Payload::Scalar(_)));
        assert_eq!(first.as_scalar().unwrap().as_f32_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn rank_two_tensor_slices_stay_tensors() {
        let parent = Payload::Tensor(
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
        );
        assert_eq!(parent.each_axis().unwrap(), (0, 2));
        let first = parent.each_slice(0).unwrap();
        assert!(matches!(first, Payload::Tensor(_)));
        assert_eq!(first.as_tensor().unwrap().shape.as_slice(), &[3]);
    }

    #[test]
    fn restack_is_driven_by_the_loop_input() {
        let parent = Payload::Tensor(
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
        );
        let items = vec![parent.each_slice(0).unwrap(), parent.each_slice(1).unwrap()];
        let stacked = parent.each_restack(&items).unwrap();
        assert!(matches!(stacked, Payload::Tensor(_)));
        assert_eq!(stacked.as_tensor().unwrap().shape.as_slice(), &[2, 3]);
        assert_eq!(
            stacked.as_tensor().unwrap().as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn restack_rejects_a_converted_body() {
        let parent = Payload::Tensor(
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
        );
        // A body that turned its rank-1 slices into rank-2 tensors no longer
        // matches the loop input, so the loop cannot be restacked.
        let items = vec![
            Payload::Tensor(Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            Payload::Tensor(Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
        ];
        let err = parent.each_restack(&items).unwrap_err();
        assert!(err.to_string().contains("rank-2 value"), "{}", err);
    }

    #[test]
    fn restack_rejects_a_mixed_body() {
        let parent = Payload::Tensor(
            Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(),
        );
        let items = vec![
            parent.each_slice(0).unwrap(),
            Payload::Data {
                buffer: RVec::new(),
            },
        ];
        let err = parent.each_restack(&items).unwrap_err();
        assert!(
            err.to_string().contains("iteration 1 returned Data"),
            "{}",
            err
        );
    }

    #[test]
    fn rank_one_tensor_slices_restack_into_rank_one() {
        let parent = Payload::Tensor(Tensor::from_f32_shape(&[1.0, 2.0, 3.0], vec![3]).unwrap());
        let items: Vec<Payload> = (0..3).map(|i| parent.each_slice(i).unwrap()).collect();
        let stacked = parent.each_restack(&items).unwrap();
        assert_eq!(stacked.as_tensor().unwrap().shape.as_slice(), &[3]);
    }

    #[test]
    fn mono_audio_slices_are_scalars_and_restack_to_audio() {
        let audio = Audio::from_f32_planar(&[1.0, 2.0, 3.0, 4.0], 1, 44100).unwrap();
        let parent = Payload::Audio(audio);
        assert_eq!(parent.each_axis().unwrap(), (0, 4));
        let items: Vec<Payload> = (0..4).map(|i| parent.each_slice(i).unwrap()).collect();
        assert!(items.iter().all(|p| matches!(p, Payload::Scalar(_))));
        let stacked = parent.each_restack(&items).unwrap();
        match stacked {
            Payload::Audio(a) => {
                assert_eq!(a.channels(), 1);
                assert_eq!(a.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected Audio, got {}", payload_kind_name(&other)),
        }
    }

    #[test]
    fn stereo_audio_slices_are_mono_audio() {
        let audio = Audio::from_f32_planar(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 48000).unwrap();
        let parent = Payload::Audio(audio);
        assert_eq!(parent.each_axis().unwrap(), (0, 2));
        let items: Vec<Payload> = (0..2).map(|i| parent.each_slice(i).unwrap()).collect();
        assert!(items.iter().all(|p| matches!(p, Payload::Audio(_))));
        let stacked = parent.each_restack(&items).unwrap();
        match stacked {
            Payload::Audio(a) => {
                assert_eq!(a.channels(), 2);
                assert_eq!(a.sample_rate, 48000);
                assert_eq!(a.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected Audio, got {}", payload_kind_name(&other)),
        }
    }

    #[test]
    fn image_slices_are_grayscale_images() {
        let image = Image::from_f32_hwc(
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
            2,
            2,
            ColorSpace::Rgb,
        )
        .unwrap();
        let parent = Payload::Image(image);
        assert_eq!(parent.each_axis().unwrap(), (2, 3));
        let items: Vec<Payload> = (0..3).map(|i| parent.each_slice(i).unwrap()).collect();
        assert!(items.iter().all(|p| matches!(p, Payload::Image(_))));
        let stacked = parent.each_restack(&items).unwrap();
        match stacked {
            Payload::Image(i) => {
                assert_eq!(i.color_space, ColorSpace::Rgb);
                assert_eq!(i.tensor.shape.as_slice(), &[2, 2, 3]);
            }
            other => panic!("expected Image, got {}", payload_kind_name(&other)),
        }
    }

    #[test]
    fn each_rejects_values_that_have_no_axis() {
        let scalar = Payload::scalar_f32(1.0);
        assert!(scalar.each_axis().is_err());
        let bytes = Payload::arg("44100");
        assert!(bytes.each_axis().is_err());
        let composite = Payload::Composite(RVec::new());
        assert!(composite.each_axis().is_err());
        assert!(composite
            .each_restack(&[Payload::Tensor(Tensor::from_f32_slice(&[1.0]))])
            .is_err());
    }
}
