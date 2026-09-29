# `squeeze`

Removes dimensions of size 1 from an input multi-dimensional tensor in O(1) zero-copy.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | *None* | Dimension index to squeeze. If omitted, all dimensions of size 1 are removed. (Positional arg 0). |

## Behavior
- Adjusts shape and strides in **O(1) time** with zero memory allocations or copying.
