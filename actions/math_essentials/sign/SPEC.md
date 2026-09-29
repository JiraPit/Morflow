# `sign`

Computes the elementwise sign of an input tensor ($-1.0$ for negative, $0.0$ for zero, $+1.0$ for positive).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW sign calculation using Rayon.
