use super::{require, OpenBlas};
use core_types::{PreparedData, Tensor, TensorDType};
use std::borrow::Cow;
use std::sync::OnceLock;

type Gemm = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *const f32,
    i32,
    *const f32,
    i32,
    f32,
    *mut f32,
    i32,
);
type Dot = unsafe extern "C" fn(i32, *const f32, i32, *const f32, i32) -> f32;
type Ger =
    unsafe extern "C" fn(i32, i32, i32, f32, *const f32, i32, *const f32, i32, *mut f32, i32);
type Getrf = unsafe extern "C" fn(i32, i32, i32, *mut f32, i32, *mut i32) -> i32;
type Getri = unsafe extern "C" fn(i32, i32, *mut f32, i32, *const i32) -> i32;
type Geqrf = unsafe extern "C" fn(i32, i32, i32, *mut f32, i32, *mut f32) -> i32;
type Orgqr = unsafe extern "C" fn(i32, i32, i32, i32, *mut f32, i32, *const f32) -> i32;
type Potrf = unsafe extern "C" fn(i32, std::ffi::c_char, i32, *mut f32, i32) -> i32;

struct Numerical {
    gemm: Option<Gemm>,
    dot: Option<Dot>,
    ger: Option<Ger>,
    getrf: Option<Getrf>,
    getri: Option<Getri>,
    geqrf: Option<Geqrf>,
    orgqr: Option<Orgqr>,
    potrf: Option<Potrf>,
}
static NUMERICAL: OnceLock<Numerical> = OnceLock::new();
fn api(b: &OpenBlas) -> &'static Numerical {
    NUMERICAL.get_or_init(|| {
        // SAFETY: these signatures are the LP64 CBLAS/LAPACKE C ABI. The
        // OpenBLAS library is retained for the lifetime of these pointers.
        unsafe {
            Numerical {
                gemm: b._library.get::<Gemm>(b"cblas_sgemm\0").ok().map(|f| *f),
                dot: b._library.get::<Dot>(b"cblas_sdot\0").ok().map(|f| *f),
                ger: b._library.get::<Ger>(b"cblas_sger\0").ok().map(|f| *f),
                getrf: b
                    ._library
                    .get::<Getrf>(b"LAPACKE_sgetrf\0")
                    .ok()
                    .map(|f| *f),
                getri: b
                    ._library
                    .get::<Getri>(b"LAPACKE_sgetri\0")
                    .ok()
                    .map(|f| *f),
                geqrf: b
                    ._library
                    .get::<Geqrf>(b"LAPACKE_sgeqrf\0")
                    .ok()
                    .map(|f| *f),
                orgqr: b
                    ._library
                    .get::<Orgqr>(b"LAPACKE_sorgqr\0")
                    .ok()
                    .map(|f| *f),
                potrf: b
                    ._library
                    .get::<Potrf>(b"LAPACKE_spotrf\0")
                    .ok()
                    .map(|f| *f),
            }
        }
    })
}
fn needed<T>(symbol: Option<T>, name: &str) -> Result<T, String> {
    symbol.ok_or_else(|| format!("The shared OpenBLAS library does not export {name}; install an OpenBLAS build with the required CBLAS/LAPACKE interface"))
}
fn int(n: usize) -> Result<i32, String> {
    i32::try_from(n).map_err(|_| "Dimension exceeds the LP64 OpenBLAS integer range".into())
}
fn values(t: &Tensor) -> Cow<'_, [f32]> {
    t.as_f32_slice()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(t.to_vec_f32()))
}
fn tensor(v: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, String> {
    Tensor::from_f32_vec(v, shape).map_err(|e| e.to_string())
}
fn status(info: i32, operation: &str) -> Result<(), String> {
    match info {
        0 => Ok(()),
        n if n < 0 => Err(format!("OpenBLAS {operation} rejected argument {}", -n)),
        n => Err(format!(
            "OpenBLAS {operation} failed at diagonal {n}: singular or not positive-definite matrix"
        )),
    }
}

struct MatrixInput<'a> {
    values: Cow<'a, [f32]>,
    trans: i32,
    lda: i32,
}
impl<'a> MatrixInput<'a> {
    fn new(t: &'a Tensor) -> Result<Self, String> {
        let rank = t.rank();
        let (rows, cols) = (t.shape[rank - 2], t.shape[rank - 1]);
        // Dense matrix views, including last-two-axis transpose views, can be
        // passed directly using the CBLAS transpose flag without materializing.
        let row_major =
            t.strides[rank - 1] == 4 && (rows <= 1 || t.strides[rank - 2] == (cols * 4) as isize);
        let col_major =
            t.strides[rank - 2] == 4 && (cols <= 1 || t.strides[rank - 1] == (rows * 4) as isize);
        let mut compact = true;
        let mut stride = rows * cols * 4;
        for axis in (0..rank - 2).rev() {
            if t.shape[axis] > 1 && t.strides[axis] != stride as isize {
                compact = false;
            }
            stride *= t.shape[axis];
        }
        if t.dtype == TensorDType::F32 && compact && (row_major || col_major) {
            let len = t.num_elements();
            if let Some(bytes) = t
                .storage
                .get(t.byte_offset..t.byte_offset.saturating_add(len.saturating_mul(4)))
            {
                if bytes.as_ptr().align_offset(4) == 0 {
                    // SAFETY: a checked, aligned storage range contains exactly
                    // len f32 values; it remains borrowed for the entire call.
                    let slice =
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), len) };
                    return Ok(Self {
                        values: Cow::Borrowed(slice),
                        trans: if row_major { 111 } else { 112 },
                        lda: int(if row_major { cols } else { rows })?,
                    });
                }
            }
        }
        Ok(Self {
            values: values(t),
            trans: 111,
            lda: int(cols)?,
        })
    }
}

pub fn matmul(a: &Tensor, b: &Tensor, prepared: &PreparedData) -> Result<Tensor, String> {
    let backend = require().map_err(|e| e.to_string())?;
    let gemm = needed(api(backend).gemm, "cblas_sgemm")?;
    let shape = prepared.output_dims().map_err(|e| e.to_string())?;
    let count = shape.iter().product::<usize>();
    let mut output = vec![0f32; count];
    if count == 0 {
        return tensor(output, shape);
    }
    let (ra, rb) = (a.rank(), b.rank());
    let (m, k, n) = (a.shape[ra - 2], a.shape[ra - 1], b.shape[rb - 1]);
    let (mi, ki, ni) = (int(m)?, int(k)?, int(n)?);
    if k == 0 {
        return tensor(output, shape);
    }
    let left = MatrixInput::new(a)?;
    let right = MatrixInput::new(b)?;
    let batch = &shape[..shape.len() - 2];
    let batch_index = |mut index: usize, input: &[usize]| {
        let mut flat = 0;
        let mut stride = 1;
        for axis in (0..batch.len()).rev() {
            let c = index % batch[axis];
            index /= batch[axis];
            if axis + input.len() >= batch.len() {
                let d = input[axis + input.len() - batch.len()];
                if d != 1 {
                    flat += c * stride;
                }
                stride *= d;
            }
        }
        flat
    };
    for (i, out) in output.chunks_mut(m * n).enumerate() {
        let ao = batch_index(i, &a.shape[..ra - 2]) * m * k;
        let bo = batch_index(i, &b.shape[..rb - 2]) * k * n;
        let av = &left.values[ao..ao + m * k];
        let bv = &right.values[bo..bo + k * n];
        // SAFETY: validated, dense matrices with checked LP64 dimensions,
        // valid leading dimensions, and a distinct m*n output buffer.
        unsafe {
            gemm(
                101,
                left.trans,
                right.trans,
                mi,
                ni,
                ki,
                1.,
                av.as_ptr(),
                left.lda,
                bv.as_ptr(),
                right.lda,
                0.,
                out.as_mut_ptr(),
                ni,
            );
        }
    }
    tensor(output, shape)
}

pub fn dot(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let backend = require().map_err(|e| e.to_string())?;
    let dot = needed(api(backend).dot, "cblas_sdot")?;
    let (a, b) = (values(a), values(b));
    if a.len() != b.len() {
        return Err("dot requires equal element counts".into());
    }
    let n = int(a.len())?;
    // SAFETY: equal-length live float slices and unit strides.
    let result = if n == 0 {
        0.
    } else {
        unsafe { dot(n, a.as_ptr(), 1, b.as_ptr(), 1) }
    };
    Ok(Tensor::from_f32_slice(&[result]))
}
pub fn outer(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let backend = require().map_err(|e| e.to_string())?;
    let ger = needed(api(backend).ger, "cblas_sger")?;
    let (a, b) = (values(a), values(b));
    let (m, n) = (int(a.len())?, int(b.len())?);
    let len = a
        .len()
        .checked_mul(b.len())
        .ok_or("Outer product size overflows")?;
    let mut output = vec![0.; len];
    // SAFETY: dense row-major output with n leading dimension; slices do not
    // overlap it. Empty arrays are handled without entering CBLAS.
    if len > 0 {
        unsafe {
            ger(
                101,
                m,
                n,
                1.,
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                output.as_mut_ptr(),
                n,
            );
        }
    }
    tensor(output, vec![a.len(), b.len()])
}

pub fn inv(t: &Tensor) -> Result<Tensor, String> {
    let backend = require().map_err(|e| e.to_string())?;
    let getrf = needed(api(backend).getrf, "LAPACKE_sgetrf")?;
    let getri = needed(api(backend).getri, "LAPACKE_sgetri")?;
    let n = t.shape[t.rank() - 1];
    let ni = int(n)?;
    let mut output = values(t).into_owned();
    if n == 0 {
        return tensor(output, t.shape.to_vec());
    }
    let mut pivots = vec![0i32; n];
    for matrix in output.chunks_mut(n * n) {
        // SAFETY: interpreting row-major A as column-major A^T lets inverse
        // overwrite the same buffer with row-major inv(A), without transpose.
        unsafe {
            status(
                getrf(102, ni, ni, matrix.as_mut_ptr(), ni, pivots.as_mut_ptr()),
                "LU factorization",
            )?;
            status(
                getri(102, ni, matrix.as_mut_ptr(), ni, pivots.as_ptr()),
                "inverse",
            )?;
        }
    }
    tensor(output, t.shape.to_vec())
}
pub fn det(t: &Tensor) -> Result<Tensor, String> {
    let backend = require().map_err(|e| e.to_string())?;
    let getrf = needed(api(backend).getrf, "LAPACKE_sgetrf")?;
    let n = t.shape[t.rank() - 1];
    let ni = int(n)?;
    let batches = t.shape[..t.rank() - 2].iter().product::<usize>();
    let input = values(t);
    let mut matrix = vec![0.; n * n];
    let mut pivots = vec![0i32; n];
    let mut output = vec![1.; batches];
    if n > 0 {
        for (batch, result) in output.iter_mut().enumerate() {
            matrix.copy_from_slice(&input[batch * n * n..(batch + 1) * n * n]);
            // SAFETY: square dense writable matrix and n pivot integers.
            let info = unsafe { getrf(102, ni, ni, matrix.as_mut_ptr(), ni, pivots.as_mut_ptr()) };
            if info < 0 {
                status(info, "determinant LU factorization")?;
            }
            if info > 0 {
                *result = 0.;
                continue;
            }
            for i in 0..n {
                *result *= matrix[i * n + i];
                if pivots[i] != (i + 1) as i32 {
                    *result = -*result;
                }
            }
        }
    }
    let shape = if t.rank() == 2 {
        Vec::new()
    } else {
        t.shape[..t.rank() - 2].to_vec()
    };
    tensor(output, shape)
}
pub fn cholesky(t: &Tensor) -> Result<Tensor, String> {
    let backend = require().map_err(|e| e.to_string())?;
    let potrf = needed(api(backend).potrf, "LAPACKE_spotrf")?;
    let n = t.shape[0];
    let ni = int(n)?;
    let mut output = values(t).into_owned();
    if n > 0 {
        // SAFETY: row-major n*n matrix. 'L' matches the basics action's lower
        // triangle convention; overwritten upper entries are cleared below.
        unsafe {
            status(
                potrf(101, b'L' as std::ffi::c_char, ni, output.as_mut_ptr(), ni),
                "Cholesky factorization",
            )?;
        }
        for i in 0..n {
            output[i * n + i + 1..(i + 1) * n].fill(0.);
        }
    }
    tensor(output, vec![n, n])
}
pub fn qr(t: &Tensor) -> Result<(Tensor, Tensor), String> {
    let backend = require().map_err(|e| e.to_string())?;
    let geqrf = needed(api(backend).geqrf, "LAPACKE_sgeqrf")?;
    let orgqr = needed(api(backend).orgqr, "LAPACKE_sorgqr")?;
    let (m, n) = (t.shape[0], t.shape[1]);
    let k = m.min(n);
    let (mi, ni, ki) = (int(m)?, int(n)?, int(k)?);
    let input = values(t);
    let mut q = vec![0.; m * n];
    let mut r = vec![0.; n * n];
    if k > 0 {
        let mut columns = vec![0.; m * n];
        for i in 0..m {
            for j in 0..n {
                columns[j * m + i] = input[i * n + j];
            }
        }
        let mut tau = vec![0.; k];
        // SAFETY: column-major m*n matrix with lda=m and min(m,n) tau values.
        unsafe {
            status(
                geqrf(102, mi, ni, columns.as_mut_ptr(), mi, tau.as_mut_ptr()),
                "QR factorization",
            )?;
        }
        for i in 0..k {
            for j in i..n {
                r[i * n + j] = columns[j * m + i];
            }
        }
        // SAFETY: orgqr generates k orthonormal columns in the first m*k entries.
        unsafe {
            status(
                orgqr(102, mi, ki, ki, columns.as_mut_ptr(), mi, tau.as_ptr()),
                "Q generation",
            )?;
        }
        for i in 0..m {
            for j in 0..k {
                q[i * n + j] = columns[j * m + i];
            }
        }
    }
    Ok((tensor(q, vec![m, n])?, tensor(r, vec![n, n])?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Shape, ValueShape};
    fn plan(shape: Vec<usize>) -> PreparedData {
        PreparedData {
            runtime: None.into(),
            output: ValueShape::tensor(Shape::new(shape)),
            args: ActionArgs::default().into(),
            fields: Default::default(),
        }
    }
    fn t(values: &[f32], shape: Vec<usize>) -> Tensor {
        Tensor::from_f32_shape(values, shape).unwrap()
    }
    fn near(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }
    #[test]
    fn gemm_broadcasts_batches_and_borrows_transpose_views() {
        let a = t(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            vec![2, 2, 3],
        );
        let b = t(&[1., 2., 3., 4., 5., 6.], vec![3, 2]);
        let out = matmul(&a, &b, &plan(vec![2, 2, 2])).unwrap();
        near(
            &out.to_vec_f32(),
            &[22., 28., 49., 64., 76., 100., 103., 136.],
        );
        let av = a.transpose(-2, -1).unwrap();
        let bv = b.transpose(0, 1).unwrap();
        // Both transpose flags, including a batch of dense column-major views.
        let out = matmul(&bv, &av, &plan(vec![2, 2, 2])).unwrap();
        near(
            &out.to_vec_f32(),
            &[22., 49., 28., 64., 76., 103., 100., 136.],
        );
        let empty = t(&[], vec![2, 0]);
        let right = t(&[], vec![0, 3]);
        near(
            &matmul(&empty, &right, &plan(vec![2, 3]))
                .unwrap()
                .to_vec_f32(),
            &[0.; 6],
        );
    }
    #[test]
    fn gemm_handles_slices_dtypes_broadcasting_and_unchanged_inputs() {
        let source = t(&[1., 9., 2., 9., 3., 9., 4., 9.], vec![2, 4]);
        let a = source.slice_range(1, 0, 4, 2).unwrap();
        let b = t(&[1., 0., 0., 1.], vec![2, 2]);
        near(
            &matmul(&a, &b, &plan(vec![2, 2])).unwrap().to_vec_f32(),
            &[1., 2., 3., 4.],
        );
        let i = a.cast(TensorDType::I32).unwrap();
        near(
            &matmul(&i, &b, &plan(vec![2, 2])).unwrap().to_vec_f32(),
            &[1., 2., 3., 4.],
        );
        near(&source.to_vec_f32(), &[1., 9., 2., 9., 3., 9., 4., 9.]);
        let left = t(&[1., 2., 3., 4.], vec![2, 1, 2, 1]);
        let right = t(&[5., 6., 7., 8., 9., 10.], vec![1, 3, 1, 2]);
        let result = matmul(&left, &right, &plan(vec![2, 3, 2, 2])).unwrap();
        near(
            &result.to_vec_f32(),
            &[
                5., 6., 10., 12., 7., 8., 14., 16., 9., 10., 18., 20., 15., 18., 20., 24., 21.,
                24., 28., 32., 27., 30., 36., 40.,
            ],
        );
    }
    #[test]
    fn inverse_and_determinant_cover_batches_singular_and_empty_matrices() {
        let a = t(&[4., 7., 2., 6., 1., 0., 0., 2.], vec![2, 2, 2]);
        near(
            &inv(&a).unwrap().to_vec_f32(),
            &[0.6, -0.7, -0.2, 0.4, 1., 0., 0., 0.5],
        );
        near(&det(&a).unwrap().to_vec_f32(), &[10., 2.]);
        let singular = t(&[1., 2., 2., 4.], vec![2, 2]);
        assert!(inv(&singular).is_err());
        near(&det(&singular).unwrap().to_vec_f32(), &[0.]);
        let empty = t(&[], vec![0, 0]);
        assert!(inv(&empty).unwrap().to_vec_f32().is_empty());
        let determinant = det(&empty).unwrap();
        assert_eq!(determinant.rank(), 0);
        near(&determinant.to_vec_f32(), &[1.]);
    }
    #[test]
    fn qr_reconstructs_tall_wide_rank_deficient_and_empty_inputs() {
        for (data, shape) in [
            (vec![1., 2., 3., 4., 5., 6.], vec![3, 2]),
            (vec![1., 2., 3., 4., 5., 6.], vec![2, 3]),
            (vec![1., 2., 2., 4., 3., 6.], vec![3, 2]),
            (vec![], vec![0, 3]),
            (vec![], vec![3, 0]),
        ] {
            let a = t(&data, shape.clone());
            let (q, r) = qr(&a).unwrap();
            assert_eq!(q.shape.as_slice(), shape);
            assert_eq!(r.shape.as_slice(), [shape[1], shape[1]]);
            let qv = q.to_vec_f32();
            let rv = r.to_vec_f32();
            let n = shape[1];
            let mut reconstructed = vec![0.; data.len()];
            for i in 0..shape[0] {
                for j in 0..n {
                    for k in 0..n {
                        reconstructed[i * n + j] += qv[i * n + k] * rv[k * n + j];
                    }
                }
            }
            near(&reconstructed, &data);
            let k = shape[0].min(n);
            for i in 0..k {
                for j in 0..k {
                    let inner = (0..shape[0])
                        .map(|row| qv[row * n + i] * qv[row * n + j])
                        .sum::<f32>();
                    near(&[inner], &[if i == j { 1. } else { 0. }]);
                }
            }
        }
    }
    #[test]
    fn cholesky_and_vector_operations_use_correct_triangle_and_empty_shapes() {
        let a = t(&[4., 2., 2., 3.], vec![2, 2]);
        near(
            &cholesky(&a).unwrap().to_vec_f32(),
            &[2., 0., 1., 2f32.sqrt()],
        );
        assert!(cholesky(&t(&[1., 2., 2., 1.], vec![2, 2])).is_err());
        assert!(cholesky(&t(&[], vec![0, 0]))
            .unwrap()
            .to_vec_f32()
            .is_empty());
        let a = t(&[1., 2., 3.], vec![3]);
        let b = t(&[4., 5., 6.], vec![3]);
        near(&dot(&a, &b).unwrap().to_vec_f32(), &[32.]);
        near(
            &outer(&a, &b).unwrap().to_vec_f32(),
            &[4., 5., 6., 8., 10., 12., 12., 15., 18.],
        );
        let empty = t(&[], vec![0]);
        near(&dot(&empty, &empty).unwrap().to_vec_f32(), &[0.]);
        assert_eq!(outer(&empty, &a).unwrap().shape.as_slice(), [0, 3]);
    }
}
