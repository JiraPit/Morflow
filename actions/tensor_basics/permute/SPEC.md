# `permute`

Reorders all dimensions of an input multi-dimensional tensor in O(1) zero-copy.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dims` | `string` | *Required* | Permutation order string, e.g. `"[2, 0, 1]"` or `"2, 0, 1"`. (Positional arg 0). |

## Behavior
- Reorders shape dimensions and corresponding strides in **O(1)** time without copying memory.
