# `div`

Performs elementwise division ($x / v$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `value` | `float` | `1.0` | Scalar divisor. (Positional arg 0). |

## Behavior
- Parallel in-place COW division using Rayon across CPU cores.
