# `permute`

Reorders all dimensions of an input multi-dimensional tensor in O(1) zero-copy.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dims` | `string` | *Required* | Permutation order string, e.g. `"[2, 0, 1]"` or `"2, 0, 1"`. (Positional arg 0). |

## Behavior
- Reorders shape dimensions and corresponding strides in **O(1)** time without copying memory.
