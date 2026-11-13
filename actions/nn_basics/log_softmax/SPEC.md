# `log_softmax`

Applies the numerically stable Log-Softmax function along a specified axis: $x_i - \max(x) - \ln\left(\sum e^{x_j - \max(x)}\right)$.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` input produces `F32`, computed in single precision exactly as before.
- `F64` input produces `F64`, computed in double precision.
- Integer input (`U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`) produces `F64`.
- NaN values in the input behave like NumPy (they propagate through the result).
- A NaN anywhere along the reduced axis makes that whole slice NaN, like NumPy.

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `-1` | Axis along which log_softmax is computed. (Positional arg 0). |

## Behavior
- Numerically stable logarithm of softmax without underflow/overflow.
