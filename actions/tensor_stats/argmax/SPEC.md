# `argmax`

Returns the indices of the maximum values along an axis.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: Input `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`; output `I32`

## DTypes
- The output is always `I32`.
- Ties pick the first matching index. A NaN counts as the extreme value, so the index of the first NaN is returned, like NumPy.

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `-1` | Axis along which argmax is computed. (Positional arg 0). |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Computes index of maximum elements along axis and returns an `I32` tensor.
