# `trace`

Computes the trace (sum of diagonal elements) of a 2D matrix or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Scalar` for a single matrix, `Payload::Tensor` for a batch)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Sums main diagonal elements $A_{ii}$ across spatial dimensions.
