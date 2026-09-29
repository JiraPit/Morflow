# `dot`

Computes the inner product between two 1D vectors ($u \cdot v = \sum u_i v_i$).

## Interface
- **Input Type**: `DataType::Composite` (`Payload::Composite([u, v])`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes dot product in parallel using Rayon reduction.
