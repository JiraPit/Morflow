# `cumsum`

Computes the cumulative sum of tensor elements along a specified axis.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `0` | Axis along which cumulative sum is computed. (Positional arg 0). |

## Behavior
- Computes prefix sum along specified dimension in parallel across other dimensions.
