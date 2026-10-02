# `qr`

Computes the QR decomposition of a 2D matrix ($A = Q R$), returning composite payload `[Q, R]`.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Composite` (`Payload::Composite([Q, R])`)
- **Supported DTypes**: F32 computation and output; other numerical inputs are converted to F32

## Parameters
None.

## Behavior
- Computes Q and R using OpenBLAS LAPACKE Householder QR. Q has shape [m,n] and R has shape [n,n]. For wide matrices, the final n−m columns of Q and rows of R are zero.

## Component shapes
For input `[M,N]`, output is `Composite([Q, R])`, where Q has shape `[M,N]` and R has shape `[N,N]`. `shapecheck` describes both components for static checking and runtime validation.

Tap the result into `$parts`, then use `$parts[0]` for Q and `$parts[1]` for R. Chained tensor indexing, such as `$parts[0][0]`, selects a row of Q.

## Ordered shape contract
`shapecheck` validates the ordered Composite inputs and predicts the output type and dimensions. Component indexes retain their original order. The engine verifies the returned payload against that prediction during execution.

## OpenBLAS backend
This action belongs to `linalg_blas/0.1.0`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.
