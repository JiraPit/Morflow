# `clamp`

Clamps all elements of an input tensor to the interval `[min, max]`.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `min` | `float` | `f32::NEG_INFINITY` | Lower clamp bound. |
| `max` | `float` | `f32::INFINITY` | Upper clamp bound. |

## Behavior
- Parallel in-place COW clamping using Rayon.
