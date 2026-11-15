# `min`

Computes the minimum value of tensor elements along a specified axis or across all elements.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The output has the same dtype as the input.
- Reducing an empty integer tensor entirely is an error (there is no identity value); empty float tensors give ±infinity as before.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | *None* | Axis along which minimum is computed. (Positional arg 0). |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Parallel reduction across CPU threads using Rayon.
