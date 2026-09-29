# `trace`

Computes the trace (sum of diagonal elements) of a 2D matrix or batched matrices.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Sums main diagonal elements $A_{ii}$ across spatial dimensions.
