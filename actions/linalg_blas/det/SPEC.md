# `det`

Computes the matrix determinant $\det(A)$ of a square 2D matrix or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Scalar` for a single matrix, `Payload::Tensor` for a batch)
- **Supported DTypes**: Input `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`; output `F32`

## Parameters
None.

## Behavior
- Computes in `F32` and returns `F32`; inputs of any other dtype are converted to `F32` first.
- Uses OpenBLAS LAPACKE LU factorization with pivoting. Batches reuse matrix and pivot storage.

## OpenBLAS backend
This action belongs to `linalg_blas/0.1.3`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.

## Runtime plugin

Declare `plugin openblas/0.1.1` in the pipeline and run `morflow prep`. These actions require the `openblas` plugin with a compatible `^0.1.0` version. The plugin loads shared system OpenBLAS during execution; shape checking remains offline. See `plugins/openblas/README.md` for system requirements.
