# `log`

Computes the elementwise natural or base-$N$ logarithm of an input tensor.

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
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `base` | `string` | `"e"` | Logarithm base: `"e"`, `"2"`, `"10"`. (Positional arg 0). |
| `eps` | `float` | `1e-8` | Small epsilon added to prevent $\log(0)$. |

## Behavior
- Parallel in-place COW logarithm calculation using Rayon.
