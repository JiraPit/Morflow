# `det`

Computes the matrix determinant $\det(A)$ of a square 2D matrix or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Scalar` for a single matrix, `Payload::Tensor` for a batch)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes determinant via LU decomposition with row pivoting in parallel using Rayon.
