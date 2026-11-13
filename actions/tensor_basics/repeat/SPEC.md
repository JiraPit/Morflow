# `repeat`

Repeats/tiles an input multi-dimensional tensor along specified dimensions.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `repeats` | `string` | *Required* | Repeat counts per dimension, e.g. `"[2, 3]"` or `"2, 3"`. (Positional arg 0). |

## Behavior
- Allocates a new tensor with repeated element tiles along each axis.
