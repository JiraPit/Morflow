# `sqrt`

Computes the elementwise square root $\sqrt{x}$ of an input tensor (clamping negative values to $0.0$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW square root computation using Rayon.
