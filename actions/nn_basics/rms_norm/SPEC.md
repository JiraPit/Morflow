# `rms_norm`

Applies Root Mean Square Normalization (RMSNorm) over the last dimension of an input tensor: $\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `eps` | `float` | `1e-5` | Small constant added to mean square for numerical stability. (Positional arg 0). |

## Behavior
- Parallel in-place COW normalization without centering mean across the trailing feature dimension.
