# `silu`

Applies the Sigmoid Linear Unit (SiLU / Swish) activation elementwise: $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` input produces `F32`, computed in single precision exactly as before.
- `F64` input produces `F64`, computed in double precision.
- Integer input (`U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`) produces `F64`.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
None.

## Behavior
- Parallel in-place COW SiLU computation using Rayon.
