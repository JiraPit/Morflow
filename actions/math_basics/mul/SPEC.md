# `mul`

Performs elementwise multiplication with a scalar value ($x \times v$).

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar | DataType::Composite` (`Payload::Tensor`, `Payload::Scalar`, or `Payload::Composite([a, b])`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor`, or `Payload::Scalar` when the input is a scalar or both components are scalars)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Two-tensor form
Given a two-component `Composite` `[a, b]` the action computes `a * b`, broadcasting like NumPy (dimensions are matched from the trailing axis, and a length-1 or missing axis stretches). Either side can be a rank-0 `Scalar`; the result is a `Scalar` only if both are. A composite with any other number of components, a component that is not a Tensor or Scalar, shapes that cannot broadcast, or a composite together with a scalar argument is an error. The shape checker predicts the broadcast shape (leaving a dimension unknown when it depends on an unknown input dimension) and rejects shapes that can never broadcast.

## DTypes
- Scalar-argument form: the output keeps the input dtype. `F32` is computed exactly as before and `F64` in double precision. On integer tensors the argument must be a whole number that fits the dtype (otherwise it is an error), and overflow wraps like NumPy.
- Two-tensor form: the same dtype stays the same. With a float involved the result is `F64` if either side is `F64`, otherwise `F32`. Two integers of the same signedness give the wider one; mixed signedness gives the smallest signed type that holds both, or `I64` if none does. Integer overflow wraps like NumPy.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `value` | `float` / `int` | `1.0` | Scalar multiplier. (Positional arg 0). Must be omitted for a Composite input. |

## Behavior
- Parallel in-place COW multiplication using Rayon across CPU cores.
