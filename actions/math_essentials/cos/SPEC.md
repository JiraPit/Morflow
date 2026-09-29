# `cos`

Computes the elementwise trigonometric cosine $\cos(x)$ of an input tensor (input values in radians).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW cosine computation using Rayon.
