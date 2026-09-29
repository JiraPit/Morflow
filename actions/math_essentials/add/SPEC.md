# `add`

Performs elementwise addition with a scalar value or secondary tensor ($x + v$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor` / `Payload::Image` / `Payload::Audio`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor` / `Payload::Image` / `Payload::Audio`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `value` | `float` | `0.0` | Scalar value added to every element. (Positional arg 0). |

## Behavior
- Parallel in-place COW mutation using Rayon across CPU cores.
