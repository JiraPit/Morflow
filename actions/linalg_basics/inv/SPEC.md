# `inv`

Computes the matrix inverse $A^{-1}$ for square non-singular matrices or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes matrix inversion using Gauss-Jordan elimination with partial pivoting in parallel across batches using Rayon.
