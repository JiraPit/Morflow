# `abs`

Computes the elementwise absolute value $|x|$ of an input tensor.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The output has the same dtype as the input.
- Integer overflow wraps like NumPy.
- NaN values in the input behave like NumPy (they propagate through the result).
- `abs` of the most negative signed integer wraps to itself (e.g. `abs(-128i8) == -128`).

## Parameters
None.

## Behavior
- Parallel in-place COW absolute value calculation using Rayon.
