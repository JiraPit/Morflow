# `neg`

Computes the elementwise negation ($-x$) of an input tensor.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW negation using Rayon.
