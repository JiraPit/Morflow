# `avg_pool2d`

Applies 2D average pooling over an input spatial tensor or image.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- `F32` input produces `F32`, computed in single precision exactly as before.
- `F64` input produces `F64`, computed in double precision.
- Integer input (`U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`) produces `F64`.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `kernel_size` / `kernel` | `int` | `2` | Spatial pooling window size. (Positional arg 0). |
| `stride` | `int` | `2` | Stride of the pooling operation. (Positional arg 1). |

## Behavior
- Downsamples spatial dimensions ($H \times W$) by taking the average within each window in parallel using Rayon.
