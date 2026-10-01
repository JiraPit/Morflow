# `gelu`

Applies the Gaussian Error Linear Unit (GELU) activation elementwise.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `approximate` | `string` | `"none"` | Approximation method: `"none"` (exact erf) or `"tanh"`. (Positional arg 0). |

## Behavior
- Parallel in-place COW GELU computation using Rayon.
