# `roll`

Circularly shifts elements of a tensor along a specified axis.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shift` / `shifts` | `int` | `0` | Number of positions elements are shifted. (Positional arg 0). |
| `axis` / `dim` | `int` | `0` | Axis along which elements are shifted. (Positional arg 1). |

## Behavior
- Slices tensor along the split boundary and concatenates parts in reverse order.
