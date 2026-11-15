# `sum`

Computes the sum of tensor elements along a specified axis or across all elements.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` stays `F32` (computed exactly as before) and `F64` stays `F64`, computed in double precision.
- Signed integers produce `I64` and unsigned integers produce `U64`; overflow wraps like NumPy.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | *None* | Axis along which sum is computed. If omitted, sums all elements. (Positional arg 0). |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Parallel reduction across CPU threads using Rayon.
