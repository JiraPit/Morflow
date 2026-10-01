# `leaky_relu`

Applies the Leaky Rectified Linear Unit activation: $x$ if $x \ge 0$, else $\alpha x$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` / `negative_slope` | `float` | `0.01` | Slope for negative inputs. (Positional arg 0). |

## Behavior
- Parallel in-place COW computation using Rayon.
