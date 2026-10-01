# `sub`

Performs elementwise subtraction ($x - v$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `value` | `float` | `0.0` | Scalar value subtracted from every element. (Positional arg 0). |

## Behavior
- Parallel in-place COW mutation using Rayon across CPU cores.
