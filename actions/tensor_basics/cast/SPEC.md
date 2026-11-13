# `cast`

Casts the numerical data type of an input tensor to a target dtype.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` | `string` | `"f32"` | Target dtype: `"f32"`, `"f64"`, `"u8"`, `"i8"`, `"i16"`, `"i32"`, `"i64"`, `"u32"`, `"u64"`, or a NumPy-style name such as `"float64"` or `"uint32"`. (Positional arg 0). |

## Behavior
- Parallel type casting using Rayon chunks across CPU threads.
- Casts between any two of the nine dtypes. Into a float, each value is rounded at most once, so integers and `F64` values keep their precision.
- Into an integer, floats are rounded to the nearest whole number (NaN becomes 0), and every value is clamped to the target's range (e.g. F32 to U8 clamps to `[0, 255]`, and `-5` cast to U32 is `0`).
- An unknown `dtype` is an error, reported by `morflow check` before the pipeline runs.
