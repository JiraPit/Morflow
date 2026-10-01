# `diag`

Extracts the diagonal vector from a 2D matrix or constructs a 2D diagonal matrix from a 1D vector.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `diagonal` / `k` | `int` | `0` | Diagonal offset (0 = main diagonal). (Positional arg 0). |

## Behavior
- If 2D matrix: returns 1D vector of diagonal elements.
- If 1D vector: returns 2D square diagonal matrix.
