# `log`

Computes the elementwise natural or base-$N$ logarithm of an input tensor.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `base` | `string` | `"e"` | Logarithm base: `"e"`, `"2"`, `"10"`. (Positional arg 0). |
| `eps` | `float` | `1e-8` | Small epsilon added to prevent $\log(0)$. |

## Behavior
- Parallel in-place COW logarithm calculation using Rayon.
