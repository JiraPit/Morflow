# `avg_pool2d`

Applies 2D average pooling over an input spatial tensor or image.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor` / `Payload::Image`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `kernel_size` / `kernel` | `int` | `2` | Spatial pooling window size. (Positional arg 0). |
| `stride` | `int` | `2` | Stride of the pooling operation. (Positional arg 1). |

## Behavior
- Downsamples spatial dimensions ($H \times W$) by taking the average within each window in parallel using Rayon.
