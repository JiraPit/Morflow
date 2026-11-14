# `relu`

Applies the Rectified Linear Unit ($\max(0, x)$) activation elementwise.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The output has the same dtype as the input.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
None.

## Behavior
- Parallel in-place COW thresholding using Rayon.
