# `softmax`

Applies the numerically stable Softmax exponent normalization along a specified axis: $\frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `-1` | Axis along which softmax is computed. (Positional arg 0). |

## Behavior
- Subtracts maximum value along axis for numerical stability to prevent float overflows.
- Computes parallel normalized probability distribution over slices using Rayon.
