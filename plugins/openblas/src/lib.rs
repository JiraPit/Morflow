//! Runtime-only OpenBLAS loading. No OpenBLAS code or link dependency is bundled.
pub mod interface;
use core_types::{RString, RVec, Tensor, TensorDType};
use libloading::Library;
use std::ffi::{c_char, CStr, OsString};
use std::sync::{Mutex, OnceLock};

type Copy32 = unsafe extern "C" fn(i32, *const f32, i32, *mut f32, i32);
type Copy64 = unsafe extern "C" fn(i32, *const f64, i32, *mut f64, i32);

pub struct OpenBlas {
    // The library must outlive every cached function pointer.
    _library: Library,
    scopy: Copy32,
    dcopy: Copy64,
}

static BACKEND: OnceLock<OpenBlas> = OnceLock::new();
static LOAD_LOCK: Mutex<()> = Mutex::new(());

/// Called exclusively from process, never from type or shape callbacks.
/// Successful loads are reused; failures are not cached.
pub fn require() -> Result<&'static OpenBlas, RString> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = LOAD_LOCK
        .lock()
        .map_err(|_| RString::from("OpenBLAS loader lock poisoned"))?;
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let override_path = std::env::var_os("MORFLOW_OPENBLAS_LIBRARY");
    let candidates: Vec<OsString> = if let Some(path) = override_path {
        vec![path]
    } else if cfg!(target_os = "windows") {
        ["libopenblas.dll", "openblas.dll"]
            .map(OsString::from)
            .to_vec()
    } else if cfg!(target_os = "macos") {
        [
            "libopenblas.dylib",
            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
        ]
        .map(OsString::from)
        .to_vec()
    } else {
        ["libopenblas.so.0", "libopenblas.so", "libopenblasp.so.0"]
            .map(OsString::from)
            .to_vec()
    };
    let mut errors = Vec::new();
    for path in candidates {
        // SAFETY: Loading a native dependency follows the platform dynamic loader.
        // The configured path is trusted executable code, just like an action binary.
        let result = unsafe { OpenBlas::load(&path) };
        match result {
            Ok(backend) => {
                let _ = BACKEND.set(backend);
                return Ok(BACKEND.get().unwrap());
            }
            Err(error) => errors.push(format!("{}: {error}", path.to_string_lossy())),
        }
    }
    Err(format!("OpenBLAS actions require a shared LP64 OpenBLAS library at execution time. Install OpenBLAS or set MORFLOW_OPENBLAS_LIBRARY to its shared library path. {}", errors.join("; ")).into())
}

impl OpenBlas {
    unsafe fn load(path: &std::ffi::OsStr) -> Result<Self, String> {
        let library = Library::new(path).map_err(|e| e.to_string())?;
        let config = library
            .get::<unsafe extern "C" fn() -> *const c_char>(b"openblas_get_config\0")
            .map_err(|e| e.to_string())?;
        let text = config();
        if text.is_null() {
            return Err("OpenBLAS returned a null build configuration".into());
        }
        let text = CStr::from_ptr(text).to_string_lossy();
        if text.contains("USE64BITINT") {
            return Err("ILP64 OpenBLAS is unsupported; select an LP64 build".into());
        }
        let scopy = *library
            .get::<Copy32>(b"cblas_scopy\0")
            .map_err(|e| e.to_string())?;
        let dcopy = *library
            .get::<Copy64>(b"cblas_dcopy\0")
            .map_err(|e| e.to_string())?;
        Ok(Self {
            _library: library,
            scopy,
            dcopy,
        })
    }

    /// Copy checked, contiguous byte ranges. Float copies use OpenBLAS; other
    /// dtypes and unaligned storage use memcpy without numerical conversion.
    fn copy(&self, source: &[u8], output: &mut [u8], dtype: TensorDType) {
        assert_eq!(source.len(), output.len());
        let size = dtype.element_size();
        if source.is_empty() {
            return;
        }
        if source.len() < 4096
            || source.as_ptr().align_offset(size) != 0
            || output.as_ptr().align_offset(size) != 0
            || !matches!(dtype, TensorDType::F32 | TensorDType::F64)
        {
            output.copy_from_slice(source);
            return;
        }
        // Chunking keeps every CBLAS length within the LP64 integer range.
        let block = (i32::MAX as usize).saturating_mul(size);
        for (src, dst) in source.chunks(block).zip(output.chunks_mut(block)) {
            let count = (src.len() / size) as i32;
            // SAFETY: slices have equal checked lengths, natural alignment, and
            // nonoverlapping storage. The retained library uses LP64 integers.
            unsafe {
                if dtype == TensorDType::F32 {
                    (self.scopy)(count, src.as_ptr().cast(), 1, dst.as_mut_ptr().cast(), 1);
                } else {
                    (self.dcopy)(count, src.as_ptr().cast(), 1, dst.as_mut_ptr().cast(), 1);
                }
            }
        }
    }

    /// Preserve contiguous views without allocation. Materialize strided views
    /// in contiguous runs, rather than making a BLAS call per element.
    pub fn contiguous(&self, tensor: &Tensor) -> Result<Tensor, RString> {
        if tensor.is_contiguous() {
            return Ok(tensor.clone());
        }
        let size = tensor.dtype.element_size();
        let count = tensor
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| RString::from("Tensor size overflows"))?;
        let bytes = count
            .checked_mul(size)
            .ok_or_else(|| RString::from("Tensor byte size overflows"))?;
        let mut output = vec![0u8; bytes];
        if count == 0 {
            return Tensor::from_rvec_u8(output.into(), tensor.shape.to_vec(), tensor.dtype);
        }
        self.copy_view(tensor, 0, &mut output)?;
        Tensor::from_rvec_u8(output.into(), tensor.shape.to_vec(), tensor.dtype)
    }

    /// Copy a logical row/block directly out of a view into its final output
    /// region. Sliced roll inputs need no intermediate materialized tensors.
    fn copy_view(
        &self,
        tensor: &Tensor,
        first_element: usize,
        output: &mut [u8],
    ) -> Result<(), RString> {
        if output.is_empty() {
            return Ok(());
        }
        let size = tensor.dtype.element_size();
        if let Some(bytes) = tensor.as_bytes() {
            let start = first_element * size;
            self.copy(&bytes[start..start + output.len()], output, tensor.dtype);
            return Ok(());
        }
        let mut run = 1usize;
        for axis in (0..tensor.rank()).rev() {
            if tensor.shape[axis] != 1 && tensor.strides[axis] as i128 != (run * size) as i128 {
                break;
            }
            run *= tensor.shape[axis];
        }
        for (flat, dst) in output.chunks_mut(run * size).enumerate() {
            let mut index = first_element + flat * run;
            let mut offset = tensor.byte_offset as i128;
            for axis in (0..tensor.rank()).rev() {
                let coordinate = index % tensor.shape[axis];
                index /= tensor.shape[axis];
                offset += coordinate as i128 * tensor.strides[axis] as i128;
            }
            let start = usize::try_from(offset)
                .map_err(|_| RString::from("Tensor view offset is invalid"))?;
            let end = start
                .checked_add(dst.len())
                .ok_or_else(|| RString::from("Tensor view range overflows"))?;
            let src = tensor
                .storage
                .get(start..end)
                .ok_or_else(|| RString::from("Tensor view exceeds storage"))?;
            self.copy(src, dst, tensor.dtype);
        }
        Ok(())
    }

    pub fn concat(&self, tensors: &[Tensor], axis: isize) -> Result<Tensor, RString> {
        let first = tensors
            .first()
            .ok_or_else(|| RString::from("concat requires tensors"))?;
        let rank = first.rank();
        let axis = core_types::contract::axis(axis, rank, false)
            .map_err(|e| RString::from(format!("{e:?}")))?;
        let mut shape = first.shape.to_vec();
        shape[axis] = 0;
        for t in tensors {
            if t.dtype != first.dtype
                || t.rank() != rank
                || t.shape
                    .iter()
                    .enumerate()
                    .any(|(i, d)| i != axis && *d != first.shape[i])
            {
                return Err(
                    "concat requires matching dtypes and non-concatenated dimensions".into(),
                );
            }
            shape[axis] = shape[axis]
                .checked_add(t.shape[axis])
                .ok_or_else(|| RString::from("Concatenated dimension overflows"))?;
        }
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| RString::from("Tensor size overflows"))?;
        let bytes = count
            .checked_mul(first.dtype.element_size())
            .ok_or_else(|| RString::from("Tensor byte size overflows"))?;
        // Small copies are dominated by call/loop overhead; reuse the native
        // path rather than forcing BLAS calls with no measurable benefit.
        if bytes <= 65536 {
            return Tensor::concat(tensors, axis as isize);
        }
        let mut output = vec![0u8; bytes];
        if bytes == 0 {
            return Tensor::from_rvec_u8(output.into(), shape, first.dtype);
        }
        let inner = shape[axis + 1..].iter().product::<usize>() * first.dtype.element_size();
        let outer = shape[..axis].iter().product::<usize>();
        let row_bytes = shape[axis] * inner;
        for row in 0..outer {
            let mut offset = row * row_bytes;
            for input in tensors {
                let len = input.shape[axis] * inner;
                self.copy_view(
                    input,
                    row * len / first.dtype.element_size(),
                    &mut output[offset..offset + len],
                )?;
                offset += len;
            }
        }
        Tensor::from_rvec_u8(RVec::from(output), shape, first.dtype)
    }
}

pub fn concat(tensors: &[Tensor], axis: isize) -> Result<Tensor, RString> {
    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }
    require()?.concat(tensors, axis)
}

mod numerical;
pub use numerical::{cholesky, det, dot, inv, matmul, outer, qr};

/// Tile in one output allocation; avoid successively concatenating each axis.
pub fn repeat(tensor: &Tensor, repeats: &[usize]) -> Result<Tensor, RString> {
    let backend = require()?;
    let rank = tensor.rank().max(repeats.len());
    let mut source_shape = vec![1; rank - tensor.rank()];
    source_shape.extend_from_slice(&tensor.shape);
    let mut shape = source_shape.clone();
    for (axis, repeat) in repeats.iter().enumerate() {
        let i = rank - repeats.len() + axis;
        shape[i] = shape[i]
            .checked_mul((*repeat).max(1))
            .ok_or_else(|| RString::from("Repeated dimension overflows"))?;
    }
    let count = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| RString::from("Repeated tensor size overflows"))?;
    let size = tensor.dtype.element_size();
    let bytes = count
        .checked_mul(size)
        .ok_or_else(|| RString::from("Repeated tensor byte size overflows"))?;
    if shape == source_shape {
        let mut view = tensor.clone();
        while view.rank() < rank {
            view = view.unsqueeze(0)?;
        }
        return Ok(view);
    }
    let mut output = vec![0u8; bytes];
    if count > 0 {
        let input = backend.contiguous(tensor)?;
        let src = input
            .as_bytes()
            .ok_or_else(|| RString::from("Invalid tensor storage"))?;
        let last = rank - 1;
        let row_bytes = source_shape[last] * size;
        for (flat, dst) in output.chunks_mut(shape[last] * size).enumerate() {
            let mut coordinate = flat;
            let mut row = 0;
            let mut stride = 1;
            for axis in (0..last).rev() {
                let c = coordinate % shape[axis];
                coordinate /= shape[axis];
                row += (c % source_shape[axis]) * stride;
                stride *= source_shape[axis];
            }
            let source = &src[row * row_bytes..(row + 1) * row_bytes];
            for block in dst.chunks_mut(row_bytes) {
                backend.copy(source, block, tensor.dtype);
            }
        }
    }
    Tensor::from_rvec_u8(output.into(), shape, tensor.dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn floats(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor::from_f32_vec((0..n).map(|i| i as f32 - 3.).collect(), shape).unwrap()
    }
    #[test]
    fn concat_matches_basics_for_views_axes_empty_and_all_dtypes() {
        let b = require().unwrap();
        let original = floats(vec![3, 4]);
        let transposed = original.transpose(0, 1).unwrap();
        let sliced = original.slice_range(1, 0, 4, 2).unwrap();
        let empty = floats(vec![0, 3]);
        for t in [&original, &transposed, &sliced, &empty] {
            for axis in [0, 1, -1] {
                for dtype in [
                    TensorDType::F32,
                    TensorDType::F64,
                    TensorDType::I32,
                    TensorDType::U8,
                ] {
                    let t = t.cast(dtype).unwrap();
                    let inputs = [t.clone(), t];
                    let result = b.concat(&inputs, axis).unwrap();
                    let expected = Tensor::concat(&inputs, axis).unwrap();
                    assert_eq!(result.shape, expected.shape);
                    assert_eq!(result.to_contiguous_bytes(), expected.to_contiguous_bytes());
                }
            }
        }
    }
    #[test]
    fn large_concat_copies_strided_views_directly_and_preserves_float_bits() {
        let b = require().unwrap();
        let source = floats(vec![257, 263]);
        let mut reversed = source.clone();
        reversed.byte_offset = 262 * 4;
        reversed.strides[1] = -4;
        for view in [
            source.clone(),
            source.transpose(0, 1).unwrap(),
            source.slice_range(1, 0, 263, 2).unwrap(),
            reversed,
        ] {
            for axis in [0, 1] {
                let inputs = [view.clone(), view.clone()];
                let actual = b.concat(&inputs, axis).unwrap();
                let expected = Tensor::concat(&inputs, axis).unwrap();
                assert_eq!(actual.shape, expected.shape);
                assert_eq!(actual.to_contiguous_bytes(), expected.to_contiguous_bytes());
            }
        }
        let bits = [0x80000000u32, 0x7fc01234, 0x7f800000, 0xff800000];
        let values = (0..32768)
            .map(|i| f32::from_bits(bits[i % 4]))
            .collect::<Vec<_>>();
        let special = Tensor::from_f32_vec(values, vec![32768]).unwrap();
        let actual = b.concat(&[special.clone(), special.clone()], 0).unwrap();
        let expected = Tensor::concat(&[special.clone(), special], 0).unwrap();
        assert_eq!(actual.to_contiguous_bytes(), expected.to_contiguous_bytes());
    }
    #[test]
    fn materializes_transposed_sliced_negative_stride_and_empty_views() {
        let b = require().unwrap();
        let t = floats(vec![3, 4]);
        let mut reversed = t.clone();
        reversed.byte_offset = 12;
        reversed.strides[1] = -4;
        for view in [
            t.transpose(0, 1).unwrap(),
            t.slice_range(1, 0, 4, 2).unwrap(),
            reversed,
            floats(vec![0, 3]).transpose(0, 1).unwrap(),
        ] {
            let expected = view.to_contiguous_bytes();
            let actual = b.contiguous(&view).unwrap();
            assert!(actual.is_contiguous());
            assert_eq!(actual.to_contiguous_bytes(), expected);
        }
    }
    #[test]
    fn repeat_matches_axis_by_axis_reference_with_padding_and_zero_factors() {
        for shape in [vec![], vec![2, 3], vec![0, 3]] {
            let source = floats(shape);
            for repeats in [vec![], vec![2], vec![2, 3], vec![2, 0, 3]] {
                let mut expected = source.clone();
                while expected.rank() < repeats.len() {
                    expected = expected.unsqueeze(0).unwrap();
                }
                let mut full = vec![1; expected.rank() - repeats.len()];
                full.extend_from_slice(&repeats);
                for (axis, n) in full.iter().enumerate() {
                    if *n > 1 {
                        expected = Tensor::concat(&vec![expected; *n], axis as isize).unwrap();
                    }
                }
                let actual = repeat(&source, &repeats).unwrap();
                assert_eq!(actual.shape, expected.shape);
                assert_eq!(actual.to_contiguous_bytes(), expected.to_contiguous_bytes());
            }
        }
    }
    #[test]
    fn missing_library_failure_is_actionable_and_retryable() {
        const FLAG: &str = "MORFLOW_TEST_MISSING_BLAS_CHILD";
        if std::env::var_os(FLAG).is_some() {
            assert!(BACKEND.get().is_none());
            let first = require().err().unwrap();
            assert!(first.contains("MORFLOW_OPENBLAS_LIBRARY"));
            std::env::remove_var("MORFLOW_OPENBLAS_LIBRARY");
            assert!(require().is_ok());
            return;
        }
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::missing_library_failure_is_actionable_and_retryable",
            ])
            .env(FLAG, "1")
            .env(
                "MORFLOW_OPENBLAS_LIBRARY",
                "/definitely/missing/openblas.so",
            )
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
    }
}

mod operations;

/// Execute an operation using validated inputs and prepared data from an action.
/// The shared SDK owns the payload storage; the returned payload owns its result.
#[no_mangle]
pub extern "C" fn morflow_openblas_process(
    operation: u32,
    payload: core_types::Payload,
    prepared: core_types::PreparedData,
) -> core_types::Payload {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        operations::dispatch(operation, payload, prepared)
    }))
    .unwrap_or_else(|_| core_types::Payload::Error("OpenBLAS plugin processing panicked".into()))
}

#[cfg(test)]
mod interface_tests {
    use super::*;
    #[test]
    fn operation_identifiers_are_explicit() {
        assert_eq!(
            [
                interface::operation!(cholesky),
                interface::operation!(det),
                interface::operation!(dot),
                interface::operation!(inv),
                interface::operation!(matmul),
                interface::operation!(outer),
                interface::operation!(qr),
                interface::operation!(concat),
                interface::operation!(repeat),
                interface::operation!(roll)
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
    }
    #[test]
    fn unknown_operation_returns_an_error_without_loading_openblas() {
        let prepared = core_types::PreparedData {
            output: core_types::ValueShape::tensor(core_types::Shape::new([0])),
            args: core_types::ActionArgs::default().into(),
            fields: Default::default(),
            runtime: None.into(),
        };
        let output = morflow_openblas_process(
            u32::MAX,
            core_types::Payload::Tensor(core_types::Tensor::from_f32_slice(&[])),
            prepared,
        );
        assert!(
            matches!(output, core_types::Payload::Error(error) if error.contains("Unknown OpenBLAS operation"))
        );
    }
}

const _: interface::NativeProcess = morflow_openblas_process;
