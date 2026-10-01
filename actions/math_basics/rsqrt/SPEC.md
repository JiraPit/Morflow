# `rsqrt`

Computes the elementwise reciprocal square root $1/\sqrt{x + \epsilon}$ of an input tensor.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `eps` | `float` | `1e-8` | Small positive constant for numerical stability. (Positional arg 0). |

## Behavior
- Parallel in-place COW reciprocal square root calculation using Rayon.
