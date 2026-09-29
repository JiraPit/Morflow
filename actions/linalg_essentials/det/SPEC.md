# `det`

Computes the matrix determinant $\det(A)$ of a square 2D matrix or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes determinant via LU decomposition with row pivoting in parallel using Rayon.
