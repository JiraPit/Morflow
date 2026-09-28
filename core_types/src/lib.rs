pub mod audio;
pub mod image;
pub mod tensor;

pub use abi_stable;
pub use abi_stable::std_types::{RBox, RString, RVec, Tuple2};
pub use abi_stable::StableAbi;

pub use audio::{Audio, AudioChannelLayout, AudioLayout};
pub use image::{ColorSpace, Image, ImageLayout};
pub use tensor::{compute_c_contiguous_strides, Tensor, TensorDType};

#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    RawBytes = 0,
    Tensor = 1,
    Composite = 2,
    Image = 3,
    Audio = 4,
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

/// Universal payload enum passed between the engine host and dynamic action plugins across FFI.
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
