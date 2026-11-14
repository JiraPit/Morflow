# `div`

Performs elementwise division ($x / v$).

## Interface
- **Input Type**: `DataType::Tensor | DataType::Scalar | DataType::Composite` (`Payload::Tensor`, `Payload::Scalar`, or `Payload::Composite([a, b])`)
- **Output Type**: `DataType::Tensor | DataType::Scalar` (`Payload::Tensor`, or `Payload::Scalar` when the input is a scalar or both components are scalars)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Two-tensor form
Given a two-component `Composite` `[a, b]` the action computes `a / b`, broadcasting like NumPy (dimensions are matched from the trailing axis, and a length-1 or missing axis stretches). Either side can be a rank-0 `Scalar`; the result is a `Scalar` only if both are. A composite with any other number of components, a component that is not a Tensor or Scalar, shapes that cannot broadcast, or a composite together with a scalar argument is an error. The shape checker predicts the broadcast shape (leaving a dimension unknown when it depends on an unknown input dimension) and rejects shapes that can never broadcast.

## DTypes
- Scalar-argument form: `F32` stays `F32` (computed exactly as before), `F64` is divided in double precision, and integer tensors produce `F64`. The value may be any number; dividing by zero is an error.
- Two-tensor form: the same dtype stays the same. With a float involved the result is `F64` if either side is `F64`, otherwise `F32`. Two integers of the same signedness give the wider one; mixed signedness gives the smallest signed type that holds both, or `I64` if none does. An integer result is turned into `F64` (true division); division by zero follows IEEE rules.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `value` / `val` / `scalar` / `by` / `amount` | `float` / `int` | `1.0` | Scalar divisor. (Positional arg 0). Must be omitted for a Composite input. |

## Behavior
- Parallel in-place COW division using Rayon across CPU cores.
