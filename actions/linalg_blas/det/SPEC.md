# `det`

Computes the matrix determinant $\det(A)$ of a square 2D matrix or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: F32 computation and output; other numerical inputs are converted to F32

## Parameters
None.

## Behavior
- Uses OpenBLAS LAPACKE LU factorization with pivoting. Batches reuse matrix and pivot storage.

## OpenBLAS backend
This action belongs to `linalg_blas/0.1.0`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.
