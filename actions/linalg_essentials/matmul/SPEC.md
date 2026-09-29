# `matmul`

Performs 2D matrix multiplication and N-D batched matrix multiplication ($C = A \cdot B$).

## Interface
- **Input Type**: `DataType::Composite` or `DataType::Tensor` (`Payload::Composite([A, B])` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Multiplies $(M \times K)$ and $(K \times N)$ matrices to produce $(M \times N)$ output in parallel across rows using Rayon.
- Supports batch dimensions $[B, M, K] \times [B, K, N] \rightarrow [B, M, N]$.
