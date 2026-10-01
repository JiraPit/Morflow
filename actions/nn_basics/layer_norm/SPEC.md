# `layer_norm`

Applies Layer Normalization over the last dimension of an input tensor: $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `eps` | `float` | `1e-5` | Small constant added to variance for numerical stability. (Positional arg 0). |

## Behavior
- Normalizes mean to 0 and variance to 1 across the trailing feature dimension in parallel using Rayon.
