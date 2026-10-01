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
        let byte_vec =
            unsafe { Vec::from_raw_parts(data.as_mut_ptr() as *mut u8, byte_len, byte_cap) };
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
        let byte_vec =
            unsafe { Vec::from_raw_parts(data.as_mut_ptr() as *mut u8, byte_len, byte_cap) };
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
        // A rank-0 tensor is a scalar and holds exactly one element, which is the
        // empty product. Returning 0 here would make scalars unstackable.
        self.shape.iter().product()
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
        if bytes.is_empty() {
            return Some(&[]);
        }
        if !bytes.len().is_multiple_of(4) || !(bytes.as_ptr() as *const f32).is_aligned() {
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
        if bytes.is_empty() {
            return Some(&[]);
        }
        if !bytes.len().is_multiple_of(4) || !(bytes.as_ptr() as *const i32).is_aligned() {
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
        if self.num_elements() == 0 {
            return &mut [];
        }
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
        if self.num_elements() == 0 {
            return &mut [];
        }
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

    /// Reshapes the tensor to a new shape with matching total element count.
    ///
    /// If the tensor is C-contiguous, this is an **O(1) zero-copy** operation.
    /// If non-contiguous, it creates a new contiguous buffer first.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, RString> {
        let total_new: usize = if new_shape.is_empty() {
            0
        } else {
            new_shape.iter().product()
        };
        let total_cur = self.num_elements();
        if total_new != total_cur {
            return Err(RString::from(format!(
                "Cannot reshape tensor of {} elements into shape {:?} with {} elements",
                total_cur, new_shape, total_new
            )));
        }

        let elem_size = self.dtype.element_size();
        let new_strides = compute_c_contiguous_strides(&new_shape, elem_size);

        if self.is_contiguous() {
            Ok(Self {
                storage: self.storage.clone(),
                byte_offset: self.byte_offset,
                shape: RVec::from(new_shape),
                strides: new_strides,
                dtype: self.dtype,
            })
        } else {
            let contig_bytes = self.to_contiguous_bytes();
            Ok(Self {
                storage: RArc::new(contig_bytes),
                byte_offset: 0,
                shape: RVec::from(new_shape),
                strides: new_strides,
                dtype: self.dtype,
            })
        }
    }

    /// Swaps two dimensions in O(1) zero-copy by adjusting shape and strides.
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Self, RString> {
        let r = self.rank();
        let d0_idx = if dim0 < 0 { dim0 + r as isize } else { dim0 };
        let d1_idx = if dim1 < 0 { dim1 + r as isize } else { dim1 };

        if d0_idx < 0 || d0_idx as usize >= r || d1_idx < 0 || d1_idx as usize >= r {
            return Err(RString::from(format!(
                "Transpose dimensions ({}, {}) out of bounds for tensor of rank {}",
                dim0, dim1, r
            )));
        }
        let d0 = d0_idx as usize;
        let d1 = d1_idx as usize;

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(d0, d1);
        new_strides.swap(d0, d1);

        Ok(Self {
            storage: self.storage.clone(),
            byte_offset: self.byte_offset,
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    /// Reorders all dimensions according to `dims` in O(1) zero-copy.
    pub fn permute(&self, dims: &[usize]) -> Result<Self, RString> {
        let r = self.rank();
        if dims.len() != r {
            return Err(RString::from(format!(
                "Permute dimensions length {} does not match tensor rank {}",
                dims.len(),
                r
            )));
        }
        let mut seen = vec![false; r];
        for &d in dims {
            if d >= r || seen[d] {
                return Err(RString::from(format!(
                    "Invalid permutation {:?} for tensor of rank {}",
                    dims, r
                )));
            }
            seen[d] = true;
        }

        let mut new_shape = Vec::with_capacity(r);
        let mut new_strides = Vec::with_capacity(r);
        for &d in dims {
            new_shape.push(self.shape[d]);
            new_strides.push(self.strides[d]);
        }

        Ok(Self {
            storage: self.storage.clone(),
            byte_offset: self.byte_offset,
            shape: RVec::from(new_shape),
            strides: RVec::from(new_strides),
            dtype: self.dtype,
        })
    }

    /// Eliminates dimension(s) of size 1 in O(1) zero-copy.
    pub fn squeeze(&self, dim: Option<isize>) -> Result<Self, RString> {
        let r = self.rank();
        if let Some(d) = dim {
            let d_idx = if d < 0 { d + r as isize } else { d };
            if d_idx < 0 || d_idx as usize >= r {
                return Err(RString::from(format!(
                    "Squeeze dimension {} out of bounds for rank {}",
                    d, r
                )));
            }
            let d = d_idx as usize;
            if self.shape[d] == 1 {
                let mut new_shape = Vec::with_capacity(r.saturating_sub(1));
                let mut new_strides = Vec::with_capacity(r.saturating_sub(1));
                for i in 0..r {
                    if i != d {
                        new_shape.push(self.shape[i]);
                        new_strides.push(self.strides[i]);
                    }
                }
                if new_shape.is_empty() {
                    new_shape.push(1);
                    new_strides.push(self.dtype.element_size() as isize);
                }
                Ok(Self {
                    storage: self.storage.clone(),
                    byte_offset: self.byte_offset,
                    shape: RVec::from(new_shape),
                    strides: RVec::from(new_strides),
                    dtype: self.dtype,
                })
            } else {
                Ok(self.clone())
            }
        } else {
            let mut new_shape = Vec::new();
            let mut new_strides = Vec::new();
            for i in 0..r {
                if self.shape[i] != 1 {
                    new_shape.push(self.shape[i]);
                    new_strides.push(self.strides[i]);
                }
            }
            if new_shape.is_empty() {
                new_shape.push(1);
                new_strides.push(self.dtype.element_size() as isize);
            }
            Ok(Self {
                storage: self.storage.clone(),
                byte_offset: self.byte_offset,
                shape: RVec::from(new_shape),
                strides: RVec::from(new_strides),
                dtype: self.dtype,
            })
        }
    }

    /// Inserts a singleton dimension (size 1) at `dim` in O(1) zero-copy.
    pub fn unsqueeze(&self, dim: isize) -> Result<Self, RString> {
        let r = self.rank();
        let d_idx = if dim < 0 { dim + (r + 1) as isize } else { dim };
        if d_idx < 0 || d_idx as usize > r {
            return Err(RString::from(format!(
                "Unsqueeze position {} out of bounds for tensor of rank {}",
                dim, r
            )));
        }
        let dim = d_idx as usize;
        let elem_size = self.dtype.element_size();
        let next_stride = if dim < r {
            self.strides[dim] * self.shape[dim] as isize
        } else {
            elem_size as isize
        };

        let mut new_shape = self.shape.to_vec();
        let mut new_strides = self.strides.to_vec();
        new_shape.insert(dim, 1);
        new_strides.insert(dim, next_stride);

        Ok(Self {
            storage: self.storage.clone(),
            byte_offset: self.byte_offset,
            shape: RVec::from(new_shape),
            strides: RVec::from(new_strides),
            dtype: self.dtype,
        })
    }

    /// Flattens a contiguous range of dimensions [start_dim, end_dim] into a single dimension.
    pub fn flatten(&self, start_dim: usize, end_dim: isize) -> Result<Self, RString> {
        let r = self.rank();
        if r == 0 {
            return Ok(self.clone());
        }
        let end_idx = if end_dim < 0 {
            (r as isize + end_dim).max(0) as usize
        } else {
            (end_dim as usize).min(r - 1)
        };

        if start_dim >= r || start_dim > end_idx {
            return Err(RString::from(format!(
                "Invalid flatten range [{}, {}] for tensor of rank {}",
                start_dim, end_idx, r
            )));
        }

        let mut new_shape = Vec::new();
        for i in 0..start_dim {
            new_shape.push(self.shape[i]);
        }
        let flat_size: usize = self.shape[start_dim..=end_idx].iter().product();
        new_shape.push(flat_size);
        for i in (end_idx + 1)..r {
            new_shape.push(self.shape[i]);
        }

        self.reshape(new_shape)
    }

    /// Casts the tensor data type to `target_dtype`.
    pub fn cast(&self, target_dtype: TensorDType) -> Result<Self, RString> {
        if self.dtype == target_dtype {
            return Ok(self.clone());
        }

        let total = self.num_elements();
        let f32_vals = self.to_vec_f32();

        match target_dtype {
            TensorDType::F32 => Self::from_f32_vec(f32_vals, self.shape.to_vec()),
            TensorDType::U8 => {
                let u8_vec: Vec<u8> = f32_vals
                    .par_iter()
                    .map(|&x| x.clamp(0.0, 255.0).round() as u8)
                    .collect();
                Self::from_rvec_u8(RVec::from(u8_vec), self.shape.to_vec(), TensorDType::U8)
            }
            TensorDType::I32 => {
                let i32_vec: Vec<i32> = f32_vals
                    .par_iter()
                    .map(|&x| x.clamp(i32::MIN as f32, i32::MAX as f32).round() as i32)
                    .collect();
                Self::from_i32_vec(i32_vec, self.shape.to_vec())
            }
            TensorDType::F64 => {
                let f64_vec: Vec<f64> = f32_vals.par_iter().map(|&x| x as f64).collect();
                let byte_len = total * 8;
                let mut bytes = vec![0u8; byte_len];
                bytes
                    .par_chunks_exact_mut(8)
                    .zip(f64_vec.par_iter())
                    .for_each(|(chunk, &val)| {
                        chunk.copy_from_slice(&val.to_ne_bytes());
                    });
                Self::from_rvec_u8(RVec::from(bytes), self.shape.to_vec(), TensorDType::F64)
            }
            TensorDType::I16 => {
                let i16_vec: Vec<i16> = f32_vals
                    .par_iter()
                    .map(|&x| x.clamp(i16::MIN as f32, i16::MAX as f32).round() as i16)
                    .collect();
                let byte_len = total * 2;
                let mut bytes = vec![0u8; byte_len];
                bytes
                    .par_chunks_exact_mut(2)
                    .zip(i16_vec.par_iter())
                    .for_each(|(chunk, &val)| {
                        chunk.copy_from_slice(&val.to_ne_bytes());
                    });
                Self::from_rvec_u8(RVec::from(bytes), self.shape.to_vec(), TensorDType::I16)
            }
            _ => Err(RString::from(format!(
                "Unsupported cast to target dtype {:?}",
                target_dtype
            ))),
        }
    }

    /// Concatenates multiple tensors along `axis`.
    pub fn concat(tensors: &[Tensor], raw_axis: isize) -> Result<Self, RString> {
        if tensors.is_empty() {
            return Err(RString::from("Cannot concatenate empty list of tensors"));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let first = &tensors[0];
        let r = first.rank();
        let ax_idx = if raw_axis < 0 {
            raw_axis + r as isize
        } else {
            raw_axis
        };
        if ax_idx < 0 || ax_idx as usize >= r {
            return Err(RString::from(format!(
                "Concat axis {} out of bounds for tensor of rank {}",
                raw_axis, r
            )));
        }
        let axis = ax_idx as usize;
        let dtype = first.dtype;
        let base_shape = first.shape.as_slice();

        let mut total_axis_len = 0usize;
        for t in tensors {
            if t.rank() != r || t.dtype != dtype {
                return Err(RString::from(
                    "All tensors in concat must have identical rank and dtype",
                ));
            }
            for (i, &base_dim) in base_shape.iter().enumerate() {
                if i != axis && t.shape[i] != base_dim {
                    return Err(RString::from(format!(
                        "Dimension mismatch along axis {}: expected {}, got {}",
                        i, base_dim, t.shape[i]
                    )));
                }
            }
            total_axis_len = total_axis_len
                .checked_add(t.shape[axis])
                .ok_or_else(|| RString::from("Concat axis length overflows"))?;
        }

        let mut out_shape = base_shape.to_vec();
        out_shape[axis] = total_axis_len;

        let outer_size: usize = out_shape[0..axis].iter().product();
        let inner_size: usize = out_shape[(axis + 1)..r].iter().product();
        let elem_size = dtype.element_size();
        let total_bytes = out_shape
            .iter()
            .try_fold(elem_size, |n, dim| n.checked_mul(*dim))
            .ok_or_else(|| RString::from("Concat output byte count overflows"))?;

        let mut out_bytes = vec![0u8; total_bytes];

        let contig_tensors: Vec<Vec<u8>> = tensors
            .iter()
            .map(|t| {
                if let Some(b) = t.as_bytes() {
                    b.to_vec()
                } else {
                    t.to_contiguous_bytes().to_vec()
                }
            })
            .collect();

        // Copy slice blocks into proper offset in output
        let mut axis_offset = 0usize;
        for (t_idx, t) in tensors.iter().enumerate() {
            let cur_axis_len = t.shape[axis];
            let t_bytes = &contig_tensors[t_idx];

            let block_size = cur_axis_len * inner_size * elem_size;
            let out_block_size = total_axis_len * inner_size * elem_size;

            for out_idx in 0..outer_size {
                let src_start = out_idx * block_size;
                let src_end = src_start + block_size;
                let dst_start = out_idx * out_block_size + axis_offset * inner_size * elem_size;
                let dst_end = dst_start + block_size;

                out_bytes[dst_start..dst_end].copy_from_slice(&t_bytes[src_start..src_end]);
            }

            axis_offset += cur_axis_len;
        }

        Self::from_rvec_u8(RVec::from(out_bytes), out_shape, dtype)
    }

    /// Converts non-contiguous or contiguous tensor into an owned Vec<f32>.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        if let Some(values) = self.as_f32_slice() {
            return values.to_vec();
        }
        let materialized;
        let bytes = match self.as_bytes() {
            Some(bytes) => bytes,
            None => {
                materialized = self.to_contiguous_bytes();
                materialized.as_slice()
            }
        };
        match self.dtype {
            TensorDType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_ne_bytes(*b))
                .collect(),
            TensorDType::F64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_ne_bytes(*b) as f32)
                .collect(),
            TensorDType::I8 => bytes.iter().map(|b| *b as i8 as f32).collect(),
            TensorDType::U8 => bytes.iter().map(|b| *b as f32).collect(),
            TensorDType::I16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_ne_bytes(*b) as f32)
                .collect(),
            TensorDType::I32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_ne_bytes(*b) as f32)
                .collect(),
            TensorDType::I64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_ne_bytes(*b) as f32)
                .collect(),
            TensorDType::U32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_ne_bytes(*b) as f32)
                .collect(),
            TensorDType::U64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_ne_bytes(*b) as f32)
                .collect(),
        }
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

/// Helper to parse a shape string like `"[2, 4]"`, `"2, 4"`, `"(2, 4)"`, or `"2, -1"` into `Vec<usize>`.
/// If `-1` is present (at most once), infers the missing dimension from `total_elements`.
pub fn parse_shape_str(s: &str, total_elements: usize) -> Result<Vec<usize>, RString> {
    let clean = s
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_start_matches('(')
        .trim_end_matches(')');
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    let parts: Vec<&str> = clean
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    let mut shape = Vec::with_capacity(parts.len());
    let mut infer_idx = None;

    for (i, p) in parts.iter().enumerate() {
        if *p == "-1" {
            if infer_idx.is_some() {
                return Err(RString::from(
                    "Only one dimension can be inferred (-1) in shape",
                ));
            }
            infer_idx = Some(i);
            shape.push(0); // placeholder
        } else {
            let val = p
                .parse::<usize>()
                .map_err(|e| RString::from(format!("Invalid dimension '{}': {}", p, e)))?;
            shape.push(val);
        }
    }

    if let Some(idx) = infer_idx {
        let known_product: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, v)| *v)
            .try_fold(1usize, |n, dim| n.checked_mul(dim))
            .ok_or_else(|| RString::from("Shape dimension product overflows"))?;
        if known_product == 0 || !total_elements.is_multiple_of(known_product) {
            return Err(RString::from(format!(
                "Cannot infer dimension -1 for total elements {} with known product {}",
                total_elements, known_product
            )));
        }
        shape[idx] = total_elements / known_product;
    }

    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_conversion_preserves_element_counts_and_handles_empty_or_unaligned_bytes() {
        let empty = Tensor::from_f32_shape(&[], vec![0]).unwrap();
        assert_eq!(empty.as_f32_slice(), Some(&[][..]));
        assert!(empty.to_vec_f32().is_empty());
        let mut empty = empty;
        assert!(empty.as_f32_slice_mut().is_empty());
        let mut empty_i32 = Tensor::from_i32_vec(Vec::new(), vec![0, 3]).unwrap();
        assert!(empty_i32.as_i32_slice_mut().is_empty());
        let integers = Tensor::from_i32_vec(vec![1, 2, 3, 4], vec![2, 2])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_eq!(integers.to_vec_f32(), [1., 3., 2., 4.]);
        let bytes = Tensor::from_rvec_u8(vec![1, 2, 3].into(), vec![3], TensorDType::U8).unwrap();
        assert_eq!(bytes.to_vec_f32(), [1., 2., 3.]);
        let mut data = vec![0];
        data.extend_from_slice(&2.5f32.to_ne_bytes());
        let mut unaligned = Tensor::from_rvec_u8(data.into(), vec![1], TensorDType::F32).unwrap();
        unaligned.byte_offset = 1;
        assert!(unaligned.as_f32_slice().is_none());
        assert_eq!(unaligned.to_vec_f32(), [2.5]);
    }

    #[test]
    fn test_tensor_reshape_permute_squeeze() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::from_f32_shape(&data, vec![2, 3, 4]).unwrap();

        // Reshape
        let reshaped = t.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.shape.as_slice(), &[6, 4]);
        assert_eq!(reshaped.as_f32_slice().unwrap()[0], 0.0);
        assert_eq!(reshaped.as_f32_slice().unwrap()[23], 23.0);

        // Transpose
        let transposed = t.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape.as_slice(), &[3, 2, 4]);

        // Permute
        let permuted = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape.as_slice(), &[4, 2, 3]);

        // Unsqueeze & Squeeze
        let unsq = t.unsqueeze(1).unwrap();
        assert_eq!(unsq.shape.as_slice(), &[2, 1, 3, 4]);
        let sq = unsq.squeeze(Some(1)).unwrap();
        assert_eq!(sq.shape.as_slice(), &[2, 3, 4]);

        // Flatten
        let flat = t.flatten(1, 2).unwrap();
        assert_eq!(flat.shape.as_slice(), &[2, 12]);

        // Parse shape
        let s = parse_shape_str("[2, -1]", 24).unwrap();
        assert_eq!(s, vec![2, 12]);
    }

    #[test]
    fn test_tensor_concat() {
        let t1 = Tensor::from_f32_shape(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::from_f32_shape(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let c0 = Tensor::concat(&[t1.clone(), t2.clone()], 0).unwrap();
        assert_eq!(c0.shape.as_slice(), &[4, 2]);
        assert_eq!(
            c0.as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );

        let c1 = Tensor::concat(&[t1, t2], 1).unwrap();
        assert_eq!(c1.shape.as_slice(), &[2, 4]);
        assert_eq!(
            c1.as_f32_slice().unwrap(),
            &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

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
