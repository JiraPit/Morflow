//! Dtype-generic numeric helpers shared by the numeric action packs.
//!
//! Everything here follows NumPy semantics unless noted otherwise:
//! integer arithmetic wraps, NaN propagates, and binary operations broadcast
//! from the trailing axis.
use crate::{RString, Tensor, TensorDType};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;

pub use num_traits;

/// Every element type a tensor can hold.
pub trait Element: Copy + Send + Sync + PartialOrd + PartialEq + Default + 'static {
    const DTYPE: TensorDType;
    const SIZE: usize;
    fn read_ne(bytes: &[u8]) -> Self;
    fn write_ne(self, out: &mut [u8]);
    fn to_f64(self) -> f64;
    /// Integer value of an integer element; floats truncate (saturating).
    fn to_i128(self) -> i128;
    /// `as`-style conversion from an integer (wrapping for integer targets).
    fn from_i128(value: i128) -> Self;
    /// `as`-style conversion from a float.
    fn from_f64(value: f64) -> Self;
}

macro_rules! element {
    ($ty:ty, $dtype:ident) => {
        impl Element for $ty {
            const DTYPE: TensorDType = TensorDType::$dtype;
            const SIZE: usize = std::mem::size_of::<$ty>();
            #[inline]
            fn read_ne(bytes: &[u8]) -> Self {
                let mut raw = [0u8; std::mem::size_of::<$ty>()];
                raw.copy_from_slice(&bytes[..std::mem::size_of::<$ty>()]);
                <$ty>::from_ne_bytes(raw)
            }
            #[inline]
            fn write_ne(self, out: &mut [u8]) {
                out[..std::mem::size_of::<$ty>()].copy_from_slice(&self.to_ne_bytes());
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn to_i128(self) -> i128 {
                self as i128
            }
            #[inline]
            fn from_i128(value: i128) -> Self {
                value as $ty
            }
            #[inline]
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    };
}
element!(u8, U8);
element!(i8, I8);
element!(i16, I16);
element!(i32, I32);
element!(i64, I64);
element!(u32, U32);
element!(u64, U64);
element!(f32, F32);
element!(f64, F64);

/// Integer element types, with NumPy's wrapping arithmetic.
pub trait IntElement: Element + Ord + std::fmt::Display {
    const SIGNED: bool;
    const MIN_I128: i128;
    const MAX_I128: i128;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_neg(self) -> Self;
    fn wrapping_abs(self) -> Self;
    /// `self ** exponent`, wrapping on overflow. `exponent` must be >= 0.
    fn wrapping_pow_u64(self, exponent: u64) -> Self {
        let mut base = self;
        let mut exponent = exponent;
        let mut acc = Self::from_i128(1);
        while exponent > 0 {
            if exponent & 1 == 1 {
                acc = acc.wrapping_mul(base);
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.wrapping_mul(base);
            }
        }
        acc
    }
    fn sign(self) -> Self {
        let zero = Self::from_i128(0);
        if self > zero {
            Self::from_i128(1)
        } else if self < zero {
            Self::from_i128(-1)
        } else {
            zero
        }
    }
    /// Converts an exact integer that is known to fit.
    fn checked_from_i128(value: i128) -> Option<Self> {
        (Self::MIN_I128..=Self::MAX_I128)
            .contains(&value)
            .then(|| Self::from_i128(value))
    }
    fn unsigned_abs_u128(self) -> u128 {
        self.to_i128().unsigned_abs()
    }
}

macro_rules! int_element {
    ($ty:ty, $signed:expr, $abs:expr) => {
        impl IntElement for $ty {
            const SIGNED: bool = $signed;
            const MIN_I128: i128 = <$ty>::MIN as i128;
            const MAX_I128: i128 = <$ty>::MAX as i128;
            #[inline]
            fn wrapping_add(self, other: Self) -> Self {
                <$ty>::wrapping_add(self, other)
            }
            #[inline]
            fn wrapping_sub(self, other: Self) -> Self {
                <$ty>::wrapping_sub(self, other)
            }
            #[inline]
            fn wrapping_mul(self, other: Self) -> Self {
                <$ty>::wrapping_mul(self, other)
            }
            #[inline]
            fn wrapping_neg(self) -> Self {
                <$ty>::wrapping_neg(self)
            }
            #[inline]
            fn wrapping_abs(self) -> Self {
                let f: fn($ty) -> $ty = $abs;
                f(self)
            }
        }
    };
}
int_element!(u8, false, |x| x);
int_element!(i8, true, |x: i8| x.wrapping_abs());
int_element!(i16, true, |x: i16| x.wrapping_abs());
int_element!(i32, true, |x: i32| x.wrapping_abs());
int_element!(i64, true, |x: i64| x.wrapping_abs());
int_element!(u32, false, |x| x);
int_element!(u64, false, |x| x);

/// Floating point element types.
pub trait FloatElement:
    Element
    + num_traits::Float
    + num_traits::FloatConst
    + std::iter::Sum
    + std::str::FromStr
    + std::fmt::Display
    + std::fmt::Debug
{
    /// `max` that propagates NaN like NumPy's `maximum`.
    #[inline]
    fn nan_max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            self.max(other)
        }
    }
    /// `min` that propagates NaN like NumPy's `minimum`.
    #[inline]
    fn nan_min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            self.min(other)
        }
    }
    /// A constant, written once per precision so neither is rounded twice.
    fn lit(single: f32, double: f64) -> Self;
}
impl FloatElement for f32 {
    #[inline]
    fn lit(single: f32, _double: f64) -> Self {
        single
    }
}
impl FloatElement for f64 {
    #[inline]
    fn lit(_single: f32, double: f64) -> Self {
        double
    }
}

/// Dispatches on a dtype, binding a type alias in each arm.
///
/// ```ignore
/// match_dtype!(tensor.dtype, int I => int_path::<I>(), float F => float_path::<F>())
/// ```
#[macro_export]
macro_rules! match_dtype {
    ($dtype:expr, int $I:ident => $int:expr, float $F:ident => $float:expr $(,)?) => {
        match $dtype {
            $crate::TensorDType::U8 => {
                #[allow(dead_code)]
                type $I = u8;
                $int
            }
            $crate::TensorDType::I8 => {
                #[allow(dead_code)]
                type $I = i8;
                $int
            }
            $crate::TensorDType::I16 => {
                #[allow(dead_code)]
                type $I = i16;
                $int
            }
            $crate::TensorDType::I32 => {
                #[allow(dead_code)]
                type $I = i32;
                $int
            }
            $crate::TensorDType::I64 => {
                #[allow(dead_code)]
                type $I = i64;
                $int
            }
            $crate::TensorDType::U32 => {
                #[allow(dead_code)]
                type $I = u32;
                $int
            }
            $crate::TensorDType::U64 => {
                #[allow(dead_code)]
                type $I = u64;
                $int
            }
            $crate::TensorDType::F32 => {
                #[allow(dead_code)]
                type $F = f32;
                $float
            }
            $crate::TensorDType::F64 => {
                #[allow(dead_code)]
                type $F = f64;
                $float
            }
        }
    };
}

pub fn is_float(dtype: TensorDType) -> bool {
    matches!(dtype, TensorDType::F32 | TensorDType::F64)
}

pub fn is_signed_int(dtype: TensorDType) -> bool {
    matches!(
        dtype,
        TensorDType::I8 | TensorDType::I16 | TensorDType::I32 | TensorDType::I64
    )
}

pub fn is_unsigned_int(dtype: TensorDType) -> bool {
    matches!(dtype, TensorDType::U8 | TensorDType::U32 | TensorDType::U64)
}

/// The dtype a naturally fractional operation produces: floats keep their
/// precision, integers become F64.
pub fn fractional_dtype(dtype: TensorDType) -> TensorDType {
    if dtype == TensorDType::F32 {
        TensorDType::F32
    } else {
        TensorDType::F64
    }
}

/// The accumulator dtype of `sum`/`cumsum`.
pub fn sum_dtype(dtype: TensorDType) -> TensorDType {
    if is_signed_int(dtype) {
        TensorDType::I64
    } else if is_unsigned_int(dtype) {
        TensorDType::U64
    } else {
        dtype
    }
}

fn int_bits(dtype: TensorDType) -> u32 {
    (dtype.element_size() * 8) as u32
}

fn signed_of_bits(bits: u32) -> Option<TensorDType> {
    match bits {
        0..=8 => Some(TensorDType::I8),
        9..=16 => Some(TensorDType::I16),
        17..=32 => Some(TensorDType::I32),
        33..=64 => Some(TensorDType::I64),
        _ => None,
    }
}

/// Result dtype of a binary operation between two tensors.
///
/// * the same dtype stays the same;
/// * with a float involved the result is F64 if either side is F64, else F32;
/// * two integers of the same signedness give the wider one;
/// * mixed signedness gives the smallest signed type holding both, or I64.
pub fn promote(a: TensorDType, b: TensorDType) -> TensorDType {
    use TensorDType::*;
    if a == b {
        return a;
    }
    if is_float(a) || is_float(b) {
        return if a == F64 || b == F64 { F64 } else { F32 };
    }
    let (a_signed, b_signed) = (is_signed_int(a), is_signed_int(b));
    if a_signed == b_signed {
        return if a.element_size() >= b.element_size() {
            a
        } else {
            b
        };
    }
    let (signed, unsigned) = if a_signed { (a, b) } else { (b, a) };
    // An unsigned type of n bits needs n + 1 bits as a signed type.
    let need = int_bits(signed).max(int_bits(unsigned) + 1);
    signed_of_bits(need).unwrap_or(I64)
}

impl Tensor {
    /// The elements in C order, if the dtype is exactly `T`.
    pub fn to_vec_of<T: Element>(&self) -> Option<Vec<T>> {
        if self.dtype != T::DTYPE {
            return None;
        }
        let materialized;
        let bytes = match self.as_bytes() {
            Some(bytes) => bytes,
            None => {
                materialized = self.to_contiguous_bytes();
                materialized.as_slice()
            }
        };
        Some(bytes.chunks_exact(T::SIZE).map(T::read_ne).collect())
    }

    /// The elements in C order converted to `T` with `as` semantics.
    pub fn to_vec_as<T: Element>(&self) -> Vec<T> {
        match_dtype!(self.dtype,
            int I => {
                // Through the exact integer value: lossless between integer
                // types that hold it, and a single rounding into floats.
                let vals = self.to_vec_of::<I>().unwrap_or_default();
                vals.into_iter().map(|v| T::from_i128(v.to_i128())).collect()
            },
            float F => {
                let vals = self.to_vec_of::<F>().unwrap_or_default();
                vals.into_iter().map(|v| T::from_f64(v.to_f64())).collect()
            }
        )
    }

    /// The elements in C order as f64 (exact for every dtype but I64/U64).
    pub fn to_vec_f64(&self) -> Vec<f64> {
        self.to_vec_as::<f64>()
    }

    /// Builds a contiguous tensor of `T` elements.
    pub fn from_vec_of<T: Element>(data: Vec<T>, shape: Vec<usize>) -> Result<Tensor, RString> {
        let count: usize = shape.iter().product();
        if data.len() != count {
            return Err(RString::from(format!(
                "Element count mismatch: vec has {}, shape requires {}",
                data.len(),
                count
            )));
        }
        let mut bytes = vec![0u8; count * T::SIZE];
        bytes
            .par_chunks_exact_mut(T::SIZE.max(1))
            .zip(data.par_iter())
            .for_each(|(chunk, value)| value.write_ne(chunk));
        Tensor::from_rvec_u8(crate::RVec::from(bytes), shape, T::DTYPE)
    }

    /// Elementwise map that keeps the element type.
    pub fn map_same<T: Element>(
        &self,
        f: impl Fn(T) -> T + Sync + Send,
    ) -> Result<Tensor, RString> {
        let mut vals = self.to_vec_of::<T>().ok_or_else(|| {
            RString::from("Tensor dtype does not match the requested element type")
        })?;
        vals.par_iter_mut().for_each(|x| *x = f(*x));
        Tensor::from_vec_of(vals, self.shape.to_vec())
    }
}

/// The values of a tensor for a naturally fractional computation: F32 stays
/// single precision and every other dtype is computed in F64.
pub enum FloatVals {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub fn float_vals(tensor: &Tensor) -> FloatVals {
    match tensor.dtype {
        TensorDType::F32 => FloatVals::F32(tensor.to_vec_of::<f32>().unwrap_or_default()),
        _ => FloatVals::F64(tensor.to_vec_f64()),
    }
}

/// An elementwise operation whose result is naturally fractional.
pub trait FloatUnary: Sync {
    fn apply<T: FloatElement>(&self, x: T) -> T;
}

/// Applies a fractional unary operation: F32 stays F32 (in place), F64 is
/// computed in double precision, and integers are computed in F64.
pub fn apply_float_unary<O: FloatUnary>(mut tensor: Tensor, op: &O) -> Result<Tensor, RString> {
    match tensor.dtype {
        TensorDType::F32 => {
            let slice = tensor.as_f32_slice_mut();
            slice.par_iter_mut().for_each(|x| *x = op.apply(*x));
            Ok(tensor)
        }
        TensorDType::F64 => tensor.map_same::<f64>(|x| op.apply(x)),
        _ => {
            let mut vals = tensor.to_vec_f64();
            vals.par_iter_mut().for_each(|x| *x = op.apply(*x));
            Tensor::from_vec_of(vals, tensor.shape.to_vec())
        }
    }
}

/// An elementwise operation that keeps the input dtype.
pub trait SameUnary: Sync {
    fn float<T: FloatElement>(&self, x: T) -> T;
    fn int<I: IntElement>(&self, x: I) -> I;
}

/// Applies a dtype-preserving unary operation; F32 is mutated in place.
pub fn apply_same_unary<O: SameUnary>(mut tensor: Tensor, op: &O) -> Result<Tensor, RString> {
    match_dtype!(tensor.dtype,
        int I => tensor.map_same::<I>(|x| op.int(x)),
        float F => {
            if F::DTYPE == TensorDType::F32 {
                let slice = tensor.as_f32_slice_mut();
                slice.par_iter_mut().for_each(|x| *x = op.float(*x));
                Ok(tensor)
            } else {
                tensor.map_same::<F>(|x| op.float(x))
            }
        }
    )
}

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// NumPy broadcasting of two concrete shapes.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for axis in 0..rank {
        let x = if axis + a.len() >= rank {
            a[axis + a.len() - rank]
        } else {
            1
        };
        let y = if axis + b.len() >= rank {
            b[axis + b.len() - rank]
        } else {
            1
        };
        out[axis] = if x == y || y == 1 {
            x
        } else if x == 1 {
            y
        } else {
            return Err(format!(
                "operands could not be broadcast together with shapes {:?} and {:?}",
                a, b
            ));
        };
    }
    Ok(out)
}

/// Element strides of `shape` aligned to `out` with zero for broadcast axes.
fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let rank = out.len();
    let mut strides = vec![0; rank];
    let mut stride = 1;
    for (i, &len) in shape.iter().enumerate().rev() {
        let axis = rank - shape.len() + i;
        strides[axis] = if len == 1 { 0 } else { stride };
        stride *= len;
    }
    strides
}

/// Combines two broadcast operands elementwise into the output shape.
pub fn broadcast_zip<A: Copy + Sync, B: Copy + Sync, O: Copy + Send + Default>(
    a: &[A],
    a_shape: &[usize],
    b: &[B],
    b_shape: &[usize],
    out_shape: &[usize],
    f: impl Fn(A, B) -> O + Sync + Send,
) -> Vec<O> {
    let count: usize = out_shape.iter().product();
    let mut out = vec![O::default(); count];
    if count == 0 {
        return out;
    }
    if a_shape == out_shape && b_shape == out_shape {
        out.par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = f(a[i], b[i]));
        return out;
    }
    let rank = out_shape.len();
    if rank == 0 {
        out[0] = f(a[0], b[0]);
        return out;
    }
    let sa = broadcast_strides(a_shape, out_shape);
    let sb = broadcast_strides(b_shape, out_shape);
    let inner = out_shape[rank - 1];
    out.par_chunks_mut(inner)
        .enumerate()
        .for_each(|(row, chunk)| {
            let (mut oa, mut ob) = (0usize, 0usize);
            let mut rem = row;
            for axis in (0..rank - 1).rev() {
                let coordinate = rem % out_shape[axis];
                rem /= out_shape[axis];
                oa += coordinate * sa[axis];
                ob += coordinate * sb[axis];
            }
            let (ia, ib) = (sa[rank - 1], sb[rank - 1]);
            for (j, slot) in chunk.iter_mut().enumerate() {
                *slot = f(a[oa + j * ia], b[ob + j * ib]);
            }
        });
    out
}

/// Reduces every lane of `vals` (C order, `shape`) along `axis` with `f`,
/// which receives the lane's elements in axis order.
pub fn reduce_axis<T, O>(
    vals: &[T],
    shape: &[usize],
    axis: usize,
    f: impl Fn(&[T]) -> O + Sync + Send,
) -> Vec<O>
where
    T: Copy + Sync + Send,
    O: Clone + Send + Default,
{
    let outer: usize = shape[..axis].iter().product();
    let len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out = vec![O::default(); outer * inner];
    if out.is_empty() {
        return out;
    }
    out.par_chunks_mut(inner)
        .enumerate()
        .for_each(|(o, chunk)| {
            let mut lane = Vec::with_capacity(len);
            for (i, slot) in chunk.iter_mut().enumerate() {
                lane.clear();
                lane.extend((0..len).map(|a| vals[(o * len + a) * inner + i]));
                *slot = f(&lane);
            }
        });
    out
}

/// Maps every lane of `vals` along `axis` to a lane of the same length.
/// `f` receives the input lane and writes the output lane.
pub fn map_axis<T, O>(
    vals: &[T],
    shape: &[usize],
    axis: usize,
    f: impl Fn(&[T], &mut [O]) + Sync + Send,
) -> Vec<O>
where
    T: Copy + Sync + Send,
    O: Copy + Send + Sync + Default,
{
    let len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut out = vec![O::default(); vals.len()];
    if out.is_empty() || len == 0 || inner == 0 {
        return out;
    }
    out.par_chunks_mut(len * inner)
        .enumerate()
        .for_each(|(o, block)| {
            let mut lane = Vec::with_capacity(len);
            let mut result = vec![O::default(); len];
            for i in 0..inner {
                lane.clear();
                lane.extend((0..len).map(|a| vals[(o * len + a) * inner + i]));
                f(&lane, &mut result);
                for (a, value) in result.iter().enumerate() {
                    block[a * inner + i] = *value;
                }
            }
        });
    out
}

/// The tensor of a `Tensor` or `Scalar` operand of a binary operation.
pub fn operand<'a>(
    payload: &'a crate::Payload,
    action: &str,
    index: usize,
) -> Result<&'a Tensor, String> {
    match payload.unwrap_payload() {
        crate::Payload::Tensor(t) | crate::Payload::Scalar(t) => Ok(t),
        other => Err(format!(
            "Action '{action}': Composite component {index} must be a Tensor or Scalar, got {}",
            crate::payload_kind_name(other)
        )),
    }
}

/// An elementwise operation between two broadcast tensors.
pub trait BinaryOp: Sync {
    /// Whether an integer result is turned into F64 (true division).
    const FRACTIONAL: bool = false;
    fn float<T: FloatElement>(&self, a: T, b: T) -> T;
    fn int<I: IntElement>(&self, a: I, b: I) -> I;
    /// Validates the integer right-hand operands before computing.
    fn check_int<I: IntElement>(&self, _rhs: &[I]) -> Result<(), String> {
        Ok(())
    }
}

/// `a <op> b` for a two-component Composite `[a, b]`, broadcasting like
/// NumPy. The result is a Scalar only if both components are Scalars.
pub fn apply_binary<O: BinaryOp>(
    items: &[crate::Payload],
    action: &str,
    op: &O,
) -> Result<crate::Payload, String> {
    if items.len() != 2 {
        return Err(format!(
            "Action '{action}' expects a Composite of exactly 2 components [a, b], got {}",
            items.len()
        ));
    }
    let (a, b) = (
        operand(&items[0], action, 0)?,
        operand(&items[1], action, 1)?,
    );
    let both_scalar = items
        .iter()
        .all(|p| matches!(p.unwrap_payload(), crate::Payload::Scalar(_)));
    let out_shape =
        broadcast_shapes(&a.shape, &b.shape).map_err(|e| format!("Action '{action}': {e}"))?;
    let tensor = binary_tensor(a, b, &out_shape, op)?;
    Ok(if both_scalar {
        crate::Payload::Scalar(tensor)
    } else {
        crate::Payload::Tensor(tensor)
    })
}

fn binary_typed<T: Element, F: Fn(T, T) -> T + Sync + Send>(
    a: &Tensor,
    b: &Tensor,
    out_shape: &[usize],
    f: F,
) -> Result<Tensor, String> {
    let (av, bv) = (a.to_vec_as::<T>(), b.to_vec_as::<T>());
    let out = broadcast_zip(&av, &a.shape, &bv, &b.shape, out_shape, f);
    Tensor::from_vec_of(out, out_shape.to_vec()).map_err(|e| e.to_string())
}

fn binary_tensor<O: BinaryOp>(
    a: &Tensor,
    b: &Tensor,
    out_shape: &[usize],
    op: &O,
) -> Result<Tensor, String> {
    let dtype = promote(a.dtype, b.dtype);
    if !is_float(dtype) && O::FRACTIONAL {
        return binary_typed::<f64, _>(a, b, out_shape, |x, y| op.float(x, y));
    }
    let empty = out_shape.iter().product::<usize>() == 0;
    match_dtype!(dtype,
        int I => {
            if !empty {
                op.check_int(&b.to_vec_as::<I>())?;
            }
            binary_typed::<I, _>(a, b, out_shape, |x, y| op.int(x, y))
        },
        float F => binary_typed::<F, _>(a, b, out_shape, |x, y| op.float(x, y)),
    )
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

/// Parses decimal text (`5`, `-3`, `4.0`, `1e3`, `2.5E+1`) into the exact
/// integer it denotes. `None` if it is not a whole number or is too large.
pub fn parse_whole(text: &str) -> Option<i128> {
    let text = text.trim();
    let (negative, rest) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    let (mantissa, exponent) = match rest.find(['e', 'E']) {
        Some(pos) => (&rest[..pos], rest[pos + 1..].parse::<i64>().ok()?),
        None => (rest, 0),
    };
    let (int_part, frac_part) = match mantissa.find('.') {
        Some(pos) => (&mantissa[..pos], &mantissa[pos + 1..]),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part
        .bytes()
        .chain(frac_part.bytes())
        .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let digits = digits.trim_start_matches('0');
    let mut scale = exponent.checked_sub(frac_part.len() as i64)?;
    let mut digits = digits.to_string();
    while scale < 0 && digits.ends_with('0') {
        digits.pop();
        scale += 1;
    }
    if digits.is_empty() {
        return Some(0);
    }
    if !(0..=40).contains(&scale) {
        return None;
    }
    let value: BigInt = digits.parse::<BigInt>().ok()? * BigInt::from(10u8).pow(scale as u32);
    let value = if negative { -value } else { value };
    value.to_i128()
}

/// Reads an integer-tensor argument: it has to be a whole number that fits
/// the dtype exactly.
pub fn int_arg<I: IntElement>(action: &str, name: &str, text: &str) -> Result<I, String> {
    parse_whole(text)
        .and_then(I::checked_from_i128)
        .ok_or_else(|| {
            format!(
                "Action '{action}': '{name}' must be a whole number that fits {:?} for an {:?} tensor, got '{text}'",
                I::DTYPE,
                I::DTYPE
            )
        })
}

// ---------------------------------------------------------------------------
// Exact integer statistics
// ---------------------------------------------------------------------------

/// Rounds `(quo + e) * 2^-shift` to the nearest f64, ties to even, where
/// `e` is in `[0, 1)` and is non-zero exactly when `sticky`.
/// `quo` must carry at least 55 significant bits unless it is exact.
fn round_scaled(quo: &BigUint, sticky: bool, shift: i64) -> f64 {
    if quo.is_zero() {
        return 0.0;
    }
    let bits = quo.bits() as i64;
    // Weight of the least significant bit we can keep.
    let drop = (bits - 53).max(shift - 1074);
    let (mut mant, round_up) = if drop <= 0 {
        (
            (quo << (-drop) as usize).to_u64().unwrap_or(u64::MAX),
            false,
        )
    } else {
        let drop_u = drop as usize;
        let mant = quo >> drop_u;
        let low = quo - (&mant << drop_u);
        let half = BigUint::one() << (drop_u - 1);
        let up = match low.cmp(&half) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => sticky || mant.is_odd(),
        };
        (mant.to_u64().unwrap_or(u64::MAX), up)
    };
    if round_up {
        mant += 1;
    }
    let mut lsb_exp = drop - shift;
    if mant == 0 {
        return 0.0;
    }
    while mant >= (1u64 << 53) {
        mant >>= 1;
        lsb_exp += 1;
    }
    if mant < (1u64 << 52) {
        // Subnormal: the least significant bit is 2^-1074.
        let mut m = mant;
        let mut e = lsb_exp;
        while e > -1074 && m < (1u64 << 52) {
            m <<= 1;
            e -= 1;
        }
        if e == -1074 && m < (1u64 << 52) {
            return f64::from_bits(m);
        }
        mant = m;
        lsb_exp = e;
    }
    let biased = lsb_exp + 52 + 1023;
    if biased >= 2047 {
        return f64::INFINITY;
    }
    if biased <= 0 {
        return f64::from_bits(mant >> (1 - biased));
    }
    f64::from_bits(((biased as u64) << 52) | (mant & ((1u64 << 52) - 1)))
}

/// Correctly rounded `num / den` for a positive denominator.
pub fn ratio_to_f64(num: &BigInt, den: &BigUint) -> f64 {
    let negative = num.sign() == Sign::Minus;
    let magnitude = num.magnitude();
    if magnitude.is_zero() {
        return 0.0;
    }
    let shift = 56 - (magnitude.bits() as i64 - den.bits() as i64);
    let (n, d) = if shift >= 0 {
        (magnitude << shift as usize, den.clone())
    } else {
        (magnitude.clone(), den << (-shift) as usize)
    };
    let (quo, rem) = n.div_rem(&d);
    let value = round_scaled(&quo, !rem.is_zero(), shift);
    if negative {
        -value
    } else {
        value
    }
}

/// Correctly rounded `(num / den) ^ (1 / root)` for non-negative operands.
pub fn root_ratio_to_f64(num: &BigUint, den: &BigUint, root: u32) -> f64 {
    if num.is_zero() {
        return 0.0;
    }
    if root == 1 {
        return ratio_to_f64(&BigInt::from(num.clone()), den);
    }
    let root_i = root as i64;
    // Choose k so the root carries at least 56 bits: bits(N) ~ 56 * root.
    let diff = num.bits() as i64 - den.bits() as i64;
    let k = (56 * root_i - diff).div_euclid(root_i) + 1;
    let scale = k * root_i;
    let (n, d) = if scale >= 0 {
        (num << scale as usize, den.clone())
    } else {
        (num.clone(), den << (-scale) as usize)
    };
    let (big_n, rem) = n.div_rem(&d);
    let r = big_n.nth_root(root);
    let exact = rem.is_zero() && r.pow(root) == big_n;
    round_scaled(&r, !exact, k)
}

/// Exact running moments of an integer sequence.
#[derive(Clone, Copy, Default, Debug)]
pub struct IntMoments {
    pub count: u64,
    pub sum: i128,
    /// Sum of squares as a 256-bit value (low, high).
    sq_lo: u128,
    sq_hi: u128,
    pub abs_max: u128,
}

impl IntMoments {
    #[inline]
    pub fn push(&mut self, value: i128) {
        self.count += 1;
        self.sum += value;
        let a = value.unsigned_abs();
        let sq = a * a;
        let (lo, carry) = self.sq_lo.overflowing_add(sq);
        self.sq_lo = lo;
        self.sq_hi += carry as u128;
        self.abs_max = self.abs_max.max(a);
    }
    fn sum_sq(&self) -> BigUint {
        (BigUint::from(self.sq_hi) << 128usize) + BigUint::from(self.sq_lo)
    }
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        if self.sum.unsigned_abs() <= (1u128 << 53) && self.count <= (1u64 << 53) {
            // Both exactly representable: IEEE division rounds correctly.
            return self.sum as f64 / self.count as f64;
        }
        ratio_to_f64(&BigInt::from(self.sum), &BigUint::from(self.count))
    }
    /// `(numerator, denominator)` of the variance with `divisor`.
    fn var_parts(&self, unbiased: bool) -> (BigUint, BigUint) {
        let n = BigUint::from(self.count);
        let s1 = BigInt::from(self.sum);
        let num = BigInt::from(&n * self.sum_sq()) - &s1 * &s1;
        let num = num.to_biguint().unwrap_or_default();
        let divisor = if unbiased && self.count > 1 {
            BigUint::from(self.count - 1)
        } else {
            n.clone()
        };
        (num, n * divisor)
    }
    pub fn var(&self, unbiased: bool) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let (num, den) = self.var_parts(unbiased);
        ratio_to_f64(&BigInt::from(num), &den)
    }
    pub fn std(&self, unbiased: bool) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let (num, den) = self.var_parts(unbiased);
        root_ratio_to_f64(&num, &den, 2)
    }
}

/// Exact `(sum |x|^p)^(1/p)` for a positive integer `p`.
pub fn int_p_norm(values: impl Iterator<Item = i128>, p: u32) -> f64 {
    let mut total = BigUint::zero();
    for value in values {
        let a = BigUint::from(value.unsigned_abs());
        total += a.pow(p);
    }
    root_ratio_to_f64(&total, &BigUint::one(), p)
}
