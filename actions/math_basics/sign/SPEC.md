# `sign`

Computes the elementwise sign of an input tensor ($-1.0$ for negative, $0.0$ for zero, $+1.0$ for positive).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The output has the same dtype as the input.
- Integer overflow wraps like NumPy.
- NaN values in the input behave like NumPy (they propagate through the result).
- Integers produce -1, 0 or 1 in their own dtype; NaN stays NaN.

## Parameters
None.

## Behavior
- Parallel in-place COW sign calculation using Rayon.
