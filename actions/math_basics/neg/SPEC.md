# `neg`

Computes the elementwise negation ($-x$) of an input tensor.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The output has the same dtype as the input.
- Integer overflow wraps like NumPy.
- NaN values in the input behave like NumPy (they propagate through the result).
- Negating an unsigned integer wraps (e.g. `-1u8 == 255`).

## Parameters
None.

## Behavior
- Parallel in-place COW negation using Rayon.
