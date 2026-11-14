# `clamp`

Clamps all elements of an input tensor to the interval `[min, max]`.

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor` or `Payload::Scalar`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The output has the same dtype as the input.
- Integer overflow wraps like NumPy.
- NaN values in the input behave like NumPy (they propagate through the result).
- On integer tensors `min` and `max` must be whole numbers that fit the dtype, otherwise it is an error. An omitted bound leaves that side unbounded.
- Clipping follows NumPy's `clip`: `min(max(x, min), max)`, so `min > max` yields `max` and a NaN bound yields NaN.

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `min` | `float` / `int` | *unbounded* | Lower clamp bound. (Positional arg 0). |
| `max` | `float` / `int` | *unbounded* | Upper clamp bound. (Positional arg 1). |

## Behavior
- Parallel in-place COW clamping using Rayon.
