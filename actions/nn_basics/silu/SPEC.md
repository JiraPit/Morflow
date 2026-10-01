# `silu`

Applies the Sigmoid Linear Unit (SiLU / Swish) activation elementwise: $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Parallel in-place COW SiLU computation using Rayon.
