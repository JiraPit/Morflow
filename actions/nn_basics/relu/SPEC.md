# `relu`

Applies the Rectified Linear Unit ($\max(0, x)$) activation elementwise.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW thresholding using Rayon.
