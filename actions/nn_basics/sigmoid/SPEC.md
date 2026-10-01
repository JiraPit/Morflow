# `sigmoid`

Applies the standard logistic sigmoid function $\frac{1}{1 + e^{-x}}$ elementwise.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW sigmoid calculation using Rayon.
