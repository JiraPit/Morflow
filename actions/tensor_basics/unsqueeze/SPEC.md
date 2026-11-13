# `unsqueeze`

Inserts a new dimension of size 1 at a specified axis in O(1) zero-copy.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `0` | Axis position where singleton dimension is inserted. (Positional arg 0). |

## Behavior
- Inserts dimension 1 into shape with appropriate byte stride in **O(1)** time without memory allocation.
