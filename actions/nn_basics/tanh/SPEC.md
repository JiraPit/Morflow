# `tanh`

Applies the Hyperbolic Tangent ($\tanh(x)$) activation elementwise.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW hyperbolic tangent computation using Rayon.
