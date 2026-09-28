use abi_stable::std_types::{RArc, RString, RVec};
use abi_stable::StableAbi;
use rayon::prelude::*;

/// Supported numerical data types for multi-dimensional tensors.
#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    U8 = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    U32 = 5,
    U64 = 6,
    F32 = 7,
    F64 = 8,
}

impl TensorDType {
    /// Byte size for each element of this data type.
    #[inline]
    pub const fn element_size(&self) -> usize {
        match self {
            TensorDType::U8 | TensorDType::I8 => 1,
            TensorDType::I16 => 2,
            TensorDType::I32 | TensorDType::U32 | TensorDType::F32 => 4,
            TensorDType::I64 | TensorDType::U64 | TensorDType::F64 => 8,
        }
    }
}

/// A high-performance, zero-copy, multi-dimensional tensor view across FFI boundaries.
///
/// Backed by a thread-safe, reference-counted storage (`RArc<RVec<u8>>`), slicing
/// operations create new offset/stride views without re-allocating or copying data bytes.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct Tensor {
    /// Shared reference-counted buffer backing this tensor and all its slice views.
    pub storage: RArc<RVec<u8>>,
    /// Starting byte offset in the underlying buffer for this view.
    pub byte_offset: usize,
    /// Dimensions/lengths along each axis.
    pub shape: RVec<usize>,
    /// Byte stride for each dimension.
    pub strides: RVec<isize>,
    /// Primitive element type.
    pub dtype: TensorDType,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    /// Creates a new contiguous tensor from an owned `RVec<u8>` and shape.
    pub fn from_rvec_u8(
        data: RVec<u8>,
        shape: Vec<usize>,
        dtype: TensorDType,
    ) -> Result<Self, RString> {
        let elem_size = dtype.element_size();
        let num_elements: usize = shape.iter().product();
        let expected_bytes = num_elements * elem_size;

        if data.len() < expected_bytes {
            return Err(RString::from(format!(
                "Buffer size {} bytes is smaller than required {} bytes for shape {:?}",
                data.len(),
                expected_bytes,
                shape
            )));
        }

        let strides = compute_c_contiguous_strides(&shape, elem_size);
        Ok(Self {
            storage: RArc::new(data),
            byte_offset: 0,
            shape: RVec::from(shape),
            strides,
            dtype,
        })
    }

    /// Creates a 1D tensor from a slice of f32 values with bulk memory copy.
    pub fn from_f32_slice(data: &[f32]) -> Self {
        let byte_len = std::mem::size_of_val(data);
        let mut bytes = Vec::with_capacity(byte_len);
        if byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    byte_len,
                );
                bytes.set_len(byte_len);
            }
        }
        let shape = vec![data.len()];
        let strides = compute_c_contiguous_strides(&shape, 4);
        Self {
            storage: RArc::new(RVec::from(bytes)),
            byte_offset: 0,
            shape: RVec::from(shape),
            strides,
            dtype: TensorDType::F32,
        }
    }

    /// Creates a multi-dimensional tensor from a slice of f32 values with specified shape.
    pub fn from_f32_shape(data: &[f32], shape: Vec<usize>) -> Result<Self, RString> {
        let num_elements: usize = shape.iter().product();
        if data.len() != num_elements {
            return Err(RString::from(format!(
                "Element count mismatch: slice has {}, shape requires {}",
                data.len(),
                num_elements
            )));
        }
        let byte_len = std::mem::size_of_val(data);
        let mut bytes = Vec::with_capacity(byte_len);
        if byte_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    byte_len,
                );
                bytes.set_len(byte_len);
            }
        }
        let strides = compute_c_contiguous_strides(&shape, 4);
        Ok(Self {
            storage: RArc::new(RVec::from(bytes)),
            byte_offset: 0,
            shape: RVec::from(shape),
            strides,
            dtype: TensorDType::F32,
        })
    }

    /// Creates a tensor by taking ownership of an owned `Vec<f32>` with zero memory copy.
    pub fn from_f32_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, RString> {
        let num_elements: usize = shape.iter().product();
        if data.len() != num_elements {
            return Err(RString::from(format!(
                "Element count mismatch: vec has {}, shape requires {}",
                data.len(),
                num_elements
            )));
        }
        let strides = compute_c_contiguous_strides(&shape, 4);
        let mut data = std::mem::ManuallyDrop::new(data);
        let byte_len = data.len() * 4;
        let byte_cap = data.capacity() * 4;
        let byte_vec = unsafe { Vec::from_raw_parts(data.as_mut_ptr() as *mut u8, byte_len, byte_cap) };
        Ok(Self {
            storage: RArc::new(RVec::from(byte_vec)),
            byte_offset: 0,
            shape: RVec::from(shape),
            strides,
            dtype: TensorDType::F32,
        })
    }

    /// Creates a tensor by taking ownership of an owned `Vec<i32>` with zero memory copy.
    pub fn from_i32_vec(data: Vec<i32>, shape: Vec<usize>) -> Result<Self, RString> {
        let num_elements: usize = shape.iter().product();
        if data.len() != num_elements {
            return Err(RString::from(format!(
                "Element count mismatch: vec has {}, shape requires {}",
                data.len(),
                num_elements
            )));
        }
        let strides = compute_c_contiguous_strides(&shape, 4);
        let mut data = std::mem::ManuallyDrop::new(data);
        let byte_len = data.len() * 4;
        let byte_cap = data.capacity() * 4;
        let byte_vec = unsafe { Vec::from_raw_parts(data.as_mut_ptr() as *mut u8, byte_len, byte_cap) };
        Ok(Self {
            storage: RArc::new(RVec::from(byte_vec)),
            byte_offset: 0,
            shape: RVec::from(shape),
            strides,
            dtype: TensorDType::I32,
        })
    }

    /// Returns the total number of elements represented by this view.
    #[inline]
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }

    /// Returns the rank (number of dimensions) of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Checks if the tensor memory layout is standard C-contiguous (row-major).
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = compute_c_contiguous_strides(&self.shape, self.dtype.element_size());
        self.strides == expected_strides
    }

    /// Slices the tensor along an axis with a range [start, end) and step.
    ///
    /// This is an **O(1) zero-copy** operation: it creates a new view into the same memory
    /// by adjusting the offset, shape, and strides, and cloning the reference counter.
    pub fn slice_range(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self, RString> {
        if axis >= self.shape.len() {
            return Err(RString::from(format!(
                "Axis {} out of bounds for tensor of rank {}",
                axis,
                self.shape.len()
            )));
        }
        let dim_len = self.shape[axis];
        let clamped_end = end.min(dim_len);
        if start > clamped_end {
            return Err(RString::from(format!(
                "Slice start {} > end {} for axis {} of length {}",
                start, clamped_end, axis, dim_len
            )));
        }
        let step = step.max(1);
        let new_len = if clamped_end > start {
            (clamped_end - start).div_ceil(step)
        } else {
            0
        };

        let axis_stride = self.strides[axis];
        let new_byte_offset = (self.byte_offset as isize + start as isize * axis_stride) as usize;

        let mut new_shape = self.shape.clone();
        new_shape[axis] = new_len;

        let mut new_strides = self.strides.clone();
        new_strides[axis] = axis_stride * step as isize;

        Ok(Self {
            storage: self.storage.clone(),
            byte_offset: new_byte_offset,
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    /// Extracts a sub-tensor slice at a specific index along an axis (reducing the rank by 1).
    ///
    /// This is an **O(1) zero-copy** operation used heavily for tensor loop unrolling (`each ($ch)`).
    pub fn slice_axis_index(&self, axis: usize, index: usize) -> Result<Self, RString> {
        if axis >= self.shape.len() {
            return Err(RString::from(format!(
                "Axis {} out of bounds for tensor of rank {}",
                axis,
                self.shape.len()
            )));
        }
        let dim_len = self.shape[axis];
        if index >= dim_len {
            return Err(RString::from(format!(
                "Index {} out of bounds for axis {} of length {}",
                index, axis, dim_len
            )));
        }

        let axis_stride = self.strides[axis];
        let new_byte_offset = (self.byte_offset as isize + index as isize * axis_stride) as usize;

        let mut new_shape = Vec::with_capacity(self.shape.len().saturating_sub(1));
        let mut new_strides = Vec::with_capacity(self.strides.len().saturating_sub(1));

        for i in 0..self.shape.len() {
            if i != axis {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }
        }

        Ok(Self {
            storage: self.storage.clone(),
            byte_offset: new_byte_offset,
            shape: RVec::from(new_shape),
            strides: RVec::from(new_strides),
            dtype: self.dtype,
        })
    }

    /// Stacks a list of tensors with identical shapes along a new dimension at `axis` using Rayon parallelism.
    pub fn stack(tensors: &[Tensor], axis: usize) -> Result<Self, RString> {
        if tensors.is_empty() {
            return Err(RString::from("Cannot stack empty list of tensors"));
        }
        let first = &tensors[0];
        let base_shape = first.shape.as_slice();
        let dtype = first.dtype;

        for t in &tensors[1..] {
            if t.shape.as_slice() != base_shape || t.dtype != dtype {
                return Err(RString::from(
                    "All tensors in stack must have identical shape and dtype",
                ));
            }
        }

        let mut new_shape = base_shape.to_vec();
        let insert_axis = axis.min(new_shape.len());
        new_shape.insert(insert_axis, tensors.len());

        let elem_size = dtype.element_size();
        let slice_bytes = base_shape.iter().product::<usize>() * elem_size;
        let total_bytes: usize = new_shape.iter().product::<usize>() * elem_size;

        // Perform parallel block copy using Rayon work-stealing chunks
        let mut combined_bytes = vec![0u8; total_bytes];
        if slice_bytes > 0 {
            combined_bytes
                .par_chunks_mut(slice_bytes)
                .zip(tensors.par_iter())
                .for_each(|(dest, src)| {
                    if let Some(bytes) = src.as_bytes() {
                        dest.copy_from_slice(bytes);
                    } else {
                        let contig = src.to_contiguous_bytes();
                        dest.copy_from_slice(contig.as_slice());
                    }
                });
        }

        let strides = compute_c_contiguous_strides(&new_shape, elem_size);
        Ok(Self {
            storage: RArc::new(RVec::from(combined_bytes)),
            byte_offset: 0,
            shape: RVec::from(new_shape),
            strides,
            dtype,
        })
    }

    /// Access contiguous byte slice if this view is contiguous in memory.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if !self.is_contiguous() {
            return None;
        }
        let total_bytes = self.num_elements() * self.dtype.element_size();
        let raw = self.storage.as_slice();
        if self.byte_offset + total_bytes <= raw.len() {
            Some(&raw[self.byte_offset..self.byte_offset + total_bytes])
        } else {
            None
        }
    }

    /// Access contiguous f32 slice if this view is contiguous and of dtype F32.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.dtype != TensorDType::F32 {
            return None;
        }
        let bytes = self.as_bytes()?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) })
    }

    /// Access contiguous u8 slice if this view is contiguous and of dtype U8.
    pub fn as_u8_slice(&self) -> Option<&[u8]> {
        if self.dtype != TensorDType::U8 {
            return None;
        }
        self.as_bytes()
    }

    /// Access contiguous i32 slice if this view is contiguous and of dtype I32.
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.dtype != TensorDType::I32 {
            return None;
        }
        let bytes = self.as_bytes()?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i32, bytes.len() / 4) })
    }

    /// Provides Copy-on-Write (COW) mutable access to contiguous F32 data.
    ///
    /// If the tensor uniquely owns its storage and is contiguous with 0 offset,
    /// it returns a mutable slice into the existing buffer (0 allocations, 0 copies).
    /// Otherwise, it re-allocates a new contiguous buffer, updates storage, and returns the slice.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        self.ensure_contiguous_storage();
        let total_bytes = self.num_elements() * self.dtype.element_size();
        let storage_mut = RArc::get_mut(&mut self.storage).unwrap();
        let slice = &mut storage_mut[self.byte_offset..self.byte_offset + total_bytes];
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f32, slice.len() / 4) }
    }

    /// Provides Copy-on-Write (COW) mutable access to contiguous U8 data.
    pub fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        self.ensure_contiguous_storage();
        let total_bytes = self.num_elements() * self.dtype.element_size();
        let storage_mut = RArc::get_mut(&mut self.storage).unwrap();
        &mut storage_mut[self.byte_offset..self.byte_offset + total_bytes]
    }

    /// Provides Copy-on-Write (COW) mutable access to contiguous I32 data.
    pub fn as_i32_slice_mut(&mut self) -> &mut [i32] {
        self.ensure_contiguous_storage();
        let total_bytes = self.num_elements() * self.dtype.element_size();
        let storage_mut = RArc::get_mut(&mut self.storage).unwrap();
        let slice = &mut storage_mut[self.byte_offset..self.byte_offset + total_bytes];
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut i32, slice.len() / 4) }
    }

    /// Ensures that this tensor has unique ownership of a C-contiguous storage buffer.
    pub fn ensure_contiguous_storage(&mut self) {
        let is_unique_and_contig = RArc::get_mut(&mut self.storage).is_some()
            && self.is_contiguous()
            && self.byte_offset == 0;

        if !is_unique_and_contig {
            let contig_bytes = self.to_contiguous_bytes();
            self.storage = RArc::new(contig_bytes);
            self.byte_offset = 0;
            self.strides = compute_c_contiguous_strides(&self.shape, self.dtype.element_size());
        }
    }

    /// Converts non-contiguous or contiguous tensor into an owned Vec<f32>.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        let bytes = self.to_contiguous_bytes();
        let samples: &[f32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
        samples.to_vec()
    }

    /// Converts non-contiguous or contiguous view into a new contiguous byte buffer.
    pub fn to_contiguous_bytes(&self) -> RVec<u8> {
        let elem_size = self.dtype.element_size();
        let total_elems = self.num_elements();
        let mut out = Vec::with_capacity(total_elems * elem_size);
        let storage_slice = self.storage.as_slice();

        fn copy_recursive(
            dim: usize,
            current_offset: isize,
            shape: &[usize],
            strides: &[isize],
            elem_size: usize,
            storage: &[u8],
            out: &mut Vec<u8>,
        ) {
            if dim == shape.len() {
                let off = current_offset as usize;
                if off + elem_size <= storage.len() {
                    out.extend_from_slice(&storage[off..off + elem_size]);
                }
                return;
            }

            let len = shape[dim];
            let stride = strides[dim];
            for i in 0..len {
                copy_recursive(
                    dim + 1,
                    current_offset + i as isize * stride,
                    shape,
                    strides,
                    elem_size,
                    storage,
                    out,
                );
            }
        }

        copy_recursive(
            0,
            self.byte_offset as isize,
            &self.shape,
            &self.strides,
            elem_size,
            storage_slice,
            &mut out,
        );

        RVec::from(out)
    }

    /// Computes the maximum absolute amplitude across all elements in parallel using Rayon.
    /// Operates with zero heap allocations on contiguous tensor views.
    pub fn peak_abs(&self) -> f64 {
        if let Some(bytes) = self.as_bytes() {
            Self::compute_peak_abs_bytes(bytes, self.dtype)
        } else {
            let bytes = self.to_contiguous_bytes();
            Self::compute_peak_abs_bytes(bytes.as_slice(), self.dtype)
        }
    }

    fn compute_peak_abs_bytes(bytes: &[u8], dtype: TensorDType) -> f64 {
        match dtype {
            TensorDType::F32 => bytes
                .par_chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()).abs() as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::F64 => bytes
                .par_chunks_exact(8)
                .map(|b| f64::from_ne_bytes(b.try_into().unwrap()).abs())
                .reduce(|| 0.0f64, f64::max),
            TensorDType::U8 => bytes
                .par_iter()
                .map(|&b| b as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::I8 => bytes
                .par_iter()
                .map(|&b| (b as i8).abs() as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::I16 => bytes
                .par_chunks_exact(2)
                .map(|b| i16::from_ne_bytes(b.try_into().unwrap()).abs() as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::I32 => bytes
                .par_chunks_exact(4)
                .map(|b| i32::from_ne_bytes(b.try_into().unwrap()).abs() as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::I64 => bytes
                .par_chunks_exact(8)
                .map(|b| (i64::from_ne_bytes(b.try_into().unwrap())).abs() as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::U32 => bytes
                .par_chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()) as f64)
                .reduce(|| 0.0f64, f64::max),
            TensorDType::U64 => bytes
                .par_chunks_exact(8)
                .map(|b| u64::from_ne_bytes(b.try_into().unwrap()) as f64)
                .reduce(|| 0.0f64, f64::max),
        }
    }

    /// Computes the Root Mean Square (RMS) energy level in parallel using Rayon.
    /// Operates with zero heap allocations on contiguous tensor views.
    pub fn rms(&self) -> f64 {
        let num_elems = self.num_elements();
        if num_elems == 0 {
            return 0.0;
        }
        let sum_sq: f64 = if let Some(bytes) = self.as_bytes() {
            Self::compute_rms_sum_sq_bytes(bytes, self.dtype)
        } else {
            let bytes = self.to_contiguous_bytes();
            Self::compute_rms_sum_sq_bytes(bytes.as_slice(), self.dtype)
        };
        (sum_sq / num_elems as f64).sqrt()
    }

    fn compute_rms_sum_sq_bytes(bytes: &[u8], dtype: TensorDType) -> f64 {
        match dtype {
            TensorDType::F32 => bytes
                .par_chunks_exact(4)
                .map(|b| {
                    let v = f32::from_ne_bytes(b.try_into().unwrap()) as f64;
                    v * v
                })
                .sum(),
            TensorDType::F64 => bytes
                .par_chunks_exact(8)
                .map(|b| {
                    let v = f64::from_ne_bytes(b.try_into().unwrap());
                    v * v
                })
                .sum(),
            _ => 0.0,
        }
    }

    /// Computes the arithmetic mean in parallel using Rayon.
    /// Operates with zero heap allocations on contiguous tensor views.
    pub fn mean(&self) -> f64 {
        let num_elems = self.num_elements();
        if num_elems == 0 {
            return 0.0;
        }
        let sum: f64 = if let Some(bytes) = self.as_bytes() {
            Self::compute_mean_sum_bytes(bytes, self.dtype)
        } else {
            let bytes = self.to_contiguous_bytes();
            Self::compute_mean_sum_bytes(bytes.as_slice(), self.dtype)
        };
        sum / num_elems as f64
    }

    fn compute_mean_sum_bytes(bytes: &[u8], dtype: TensorDType) -> f64 {
        match dtype {
            TensorDType::F32 => bytes
                .par_chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()) as f64)
                .sum(),
            TensorDType::F64 => bytes
                .par_chunks_exact(8)
                .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                .sum(),
            _ => 0.0,
        }
    }
}

/// Helper to compute C-contiguous (row-major) byte strides for a given shape and element size.
pub fn compute_c_contiguous_strides(shape: &[usize], elem_size: usize) -> RVec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut current_stride = elem_size as isize;
    for (i, &dim) in shape.iter().enumerate().rev() {
        strides[i] = current_stride;
        current_stride *= dim as isize;
    }
    RVec::from(strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zero_copy_slicing() {
        let data: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&data, vec![4, 8]).unwrap();

        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.num_elements(), 32);
        assert!(tensor.is_contiguous());

        let row_slice = tensor.slice_range(0, 1, 3, 1).unwrap();
        assert_eq!(row_slice.shape.as_slice(), &[2, 8]);
        assert_eq!(row_slice.num_elements(), 16);
        assert_eq!(tensor.storage.as_ptr(), row_slice.storage.as_ptr());

        let row2 = tensor.slice_axis_index(0, 2).unwrap();
        assert_eq!(row2.rank(), 1);
        assert_eq!(row2.shape.as_slice(), &[8]);
        assert_eq!(row2.storage.as_ptr(), tensor.storage.as_ptr());

        let contiguous_bytes = row2.to_contiguous_bytes();
        let f32_vals: Vec<f32> = contiguous_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .collect();
        assert_eq!(
            f32_vals,
            vec![16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn test_tensor_stack_and_metrics() {
        let row1 = Tensor::from_f32_slice(&[1.0, -2.0, 3.0]);
        let row2 = Tensor::from_f32_slice(&[4.0, 0.0, -1.0]);

        assert_eq!(row1.peak_abs(), 3.0);
        assert!((row1.mean() - (2.0 / 3.0)).abs() < 1e-5);
        assert!((row1.rms() - ((14.0 / 3.0f64).sqrt())).abs() < 1e-5);

        let stacked = Tensor::stack(&[row1, row2], 0).unwrap();
        assert_eq!(stacked.shape.as_slice(), &[2, 3]);
        assert_eq!(stacked.peak_abs(), 4.0);
    }

    #[test]
    fn test_non_contiguous_tensor_metrics() {
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&data, vec![4, 4]).unwrap();

        // Extract column 1 (strided / non-contiguous): values [1.0, 5.0, 9.0, 13.0]
        let col1 = tensor.slice_axis_index(1, 1).unwrap();
        assert!(!col1.is_contiguous());
        assert_eq!(col1.peak_abs(), 13.0);
        assert_eq!(col1.mean(), (1.0 + 5.0 + 9.0 + 13.0) / 4.0);
    }

    #[test]
    fn test_empty_tensor_metrics() {
        let empty = Tensor::from_f32_slice(&[]);
        assert_eq!(empty.num_elements(), 0);
        assert_eq!(empty.peak_abs(), 0.0);
        assert_eq!(empty.mean(), 0.0);
        assert_eq!(empty.rms(), 0.0);
    }

    #[test]
    fn test_tensor_from_vec_f32_zero_copy() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32_vec(data, vec![2, 2]).unwrap();
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_cow_in_place_mutation() {
        let data = vec![10.0f32, 20.0, 30.0];
        let mut tensor = Tensor::from_f32_vec(data, vec![3]).unwrap();
        let original_ptr = tensor.storage.as_ptr();

        // Unique owner: mutates directly in-place without reallocation
        let slice = tensor.as_f32_slice_mut();
        slice[0] = 99.0;
        assert_eq!(tensor.storage.as_ptr(), original_ptr);
        assert_eq!(tensor.as_f32_slice().unwrap(), &[99.0, 20.0, 30.0]);

        // Cloned view: COW triggers reallocation on mutation
        let cloned_view = tensor.clone();
        assert_eq!(tensor.storage.as_ptr(), cloned_view.storage.as_ptr());

        let slice_mut = tensor.as_f32_slice_mut();
        slice_mut[1] = 88.0;
        // Storage pointer changed due to COW
        assert_ne!(tensor.storage.as_ptr(), cloned_view.storage.as_ptr());
        assert_eq!(tensor.as_f32_slice().unwrap(), &[99.0, 88.0, 30.0]);
        // Cloned view preserved original values
        assert_eq!(cloned_view.as_f32_slice().unwrap(), &[99.0, 20.0, 30.0]);
    }
}
