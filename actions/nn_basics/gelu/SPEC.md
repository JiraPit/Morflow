# `gelu`

Applies the Gaussian Error Linear Unit (GELU) activation elementwise.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` input produces `F32`, computed in single precision exactly as before.
- `F64` input produces `F64`, computed in double precision.
- Integer input (`U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`) produces `F64`.
- NaN values in the input behave like NumPy (they propagate through the result).
- `F64` uses an accurate double-precision `erf`; `F32` keeps its single-precision approximation.

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `approximate` | `string` | `"none"` | Approximation method: `"none"` (exact erf) or `"tanh"`. (Positional arg 0). |

## Behavior
- Parallel in-place COW GELU computation using Rayon.
