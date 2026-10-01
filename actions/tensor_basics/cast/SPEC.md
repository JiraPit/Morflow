# `cast`

Casts the numerical data type of an input tensor to a target dtype.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` | `string` | `"f32"` | Target dtype: `"f32"`, `"f64"`, `"u8"`, `"i16"`, `"i32"`, `"i64"`. (Positional arg 0). |

## Behavior
- Parallel type casting using Rayon chunks across CPU threads.
- Clamps values appropriately when downcasting (e.g. F32 to U8 clamps to `[0, 255]`).
