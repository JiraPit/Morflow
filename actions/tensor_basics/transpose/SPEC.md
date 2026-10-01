# `transpose`

Swaps two dimensions of an input tensor in O(1) zero-copy.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dim0` | `int` | `0` | First dimension index to swap. (Positional arg 0). |
| `dim1` | `int` | `1` | Second dimension index to swap. (Positional arg 1). |

## Behavior
- Adjusts strides and shape to transpose axes in **O(1) time** with zero memory allocations or byte copying.
