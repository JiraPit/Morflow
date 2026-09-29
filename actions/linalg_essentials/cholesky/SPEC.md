# `cholesky`

Computes the Cholesky factorization of a symmetric positive-definite 2D matrix ($A = L L^T$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes lower triangular matrix $L$ such that $L L^T = A$.
