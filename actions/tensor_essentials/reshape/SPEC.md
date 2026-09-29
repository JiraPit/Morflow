# `reshape`

Reshapes an input multi-dimensional tensor to a specified target shape without altering element order.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All (`F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`)

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shape` | `string` | *Required* | Target shape string, e.g. `"[4, 16]"`, `"4, 16"`, or `"[2, -1]"` (inferred dimension). (Positional arg 0). |

## Behavior
- If the tensor is contiguous in memory, performs an **O(1) zero-copy** reshape by calculating new strides.
- If non-contiguous, re-allocates a contiguous buffer before applying the shape transformation.
- Supports single inferred dimension `-1` automatically computed from the total element count.
