pub mod audio;
pub mod image;
pub mod tensor;

pub use abi_stable;
pub use abi_stable::std_types::{RBox, RString, RVec, Tuple2};
pub use abi_stable::StableAbi;

pub use audio::{Audio, AudioChannelLayout, AudioLayout};
pub use image::{ColorSpace, Image, ImageLayout};
pub use tensor::{compute_c_contiguous_strides, parse_shape_str, Tensor, TensorDType};

#[repr(transparent)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DataType(pub u8);

#[allow(non_upper_case_globals)]
impl DataType {
    pub const RawBytes: DataType = DataType(1 << 0);
    pub const Tensor: DataType = DataType(1 << 1);
    pub const Composite: DataType = DataType(1 << 2);
    pub const Image: DataType = DataType(1 << 3);
    pub const Audio: DataType = DataType(1 << 4);
    pub const Any: DataType = DataType(0b00011111);

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
        if self.contains(Self::RawBytes) {
            parts.push("RawBytes");
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
}

impl Payload {
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
}

pub type ProcessFn = extern "C" fn(Payload) -> Payload;
pub type GetTypeFn = extern "C" fn() -> DataType;
