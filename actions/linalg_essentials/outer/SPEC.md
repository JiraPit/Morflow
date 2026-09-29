# `outer`

Computes the outer product between two 1D vectors ($u \otimes v = u v^T$, shape $[M, N]$).

## Interface
- **Input Type**: `DataType::Composite` (`Payload::Composite([u, v])`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes $M \times N$ outer product matrix in parallel using Rayon.
