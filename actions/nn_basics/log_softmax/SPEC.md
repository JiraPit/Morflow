# `log_softmax`

Applies the numerically stable Log-Softmax function along a specified axis: $x_i - \max(x) - \ln\left(\sum e^{x_j - \max(x)}\right)$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `-1` | Axis along which log_softmax is computed. (Positional arg 0). |

## Behavior
- Numerically stable logarithm of softmax without underflow/overflow.
