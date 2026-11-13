# `mean`

Computes the arithmetic mean of tensor elements along a specified axis or across all elements.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` input produces `F32`, computed in single precision exactly as before.
- `F64` input produces `F64`, computed in double precision.
- Integer input (`U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`) produces `F64`. The statistic is computed exactly and only the result is rounded to `F64`.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | *None* | Axis along which mean is computed. If omitted, computes mean over all elements. (Positional arg 0). |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Parallel reduction and division across CPU threads using Rayon.
