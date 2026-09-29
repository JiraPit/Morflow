# `mul`

Performs elementwise multiplication with a scalar value ($x \times v$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `value` | `float` | `1.0` | Scalar multiplier. (Positional arg 0). |

## Behavior
- Parallel in-place COW multiplication using Rayon across CPU cores.
