# `norm`

Computes the matrix or vector $p$-norm ($L_1$, $L_2$ Euclidean, or $L_\infty$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` input produces `F32`, computed in single precision exactly as before.
- `F64` input produces `F64`, computed in double precision.
- Integer input (`U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`) produces `F64`. The statistic is computed exactly and only the result is rounded to `F64` (for `norm`: the infinity norm and positive integer orders `p`; other orders are computed in `F64`).
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `p` | `float` / `string` | `2.0` | Order of the norm (`1.0`, `2.0`, `"inf"`). (Positional arg 0). |
| `axis` / `dim` | `int` | *None* | Axis along which norm is computed. |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Parallel norm calculation across CPU threads using Rayon.
