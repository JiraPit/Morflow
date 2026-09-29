# `concat`

Concatenates multiple tensors along a specified axis.

## Interface
- **Input Type**: `DataType::Composite` or `DataType::Tensor` (`Payload::Composite([Tensor, ...])` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `0` | Axis along which tensors are concatenated. (Positional arg 0). |

## Behavior
- Verifies rank and dimensional alignment across non-concatenated axes.
- Performs parallel block copying into the resulting combined contiguous tensor buffer.
