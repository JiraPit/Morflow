# `cosine_similarity`

Computes the cosine similarity between two tensors along a specified axis: $\frac{A \cdot B}{\|A\|_2 \|B\|_2}$.

## Interface
- **Input Type**: `DataType::Composite` (`Payload::Composite([Tensor, Tensor])`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `eps` | `float` | `1e-8` | Small epsilon constant to prevent division by zero. |

## Behavior
- Computes normalized dot products between paired vectors in parallel using Rayon.
