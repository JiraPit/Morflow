# `matmul`

Performs 2D matrix multiplication and N-D batched matrix multiplication ($C = A \cdot B$).

## Interface
- **Input Type**: `DataType::Composite` or `DataType::Tensor` (`Payload::Composite([A, B])` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: F32 computation and output; other numerical inputs are converted to F32

## Parameters
None.

## Behavior
- Uses OpenBLAS `cblas_sgemm` to multiply [M,K] and [K,N] matrices into [M,N]. Contiguous F32 matrices and dense transpose views are borrowed without copying.
- Supports NumPy-style broadcasting of batch dimensions.

## Ordered shape contract
`shapecheck` validates the ordered Composite inputs and predicts the output type and dimensions. Component indexes retain their original order. The engine verifies the returned payload against that prediction during execution.

## OpenBLAS backend
This action belongs to `linalg_blas/0.1.0`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.
