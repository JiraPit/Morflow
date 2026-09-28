# `crop`

Spatial bounding box cropping and sub-region extraction for 2D and 3D tensors.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`, `I32`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `x` | `int` | `0` | Left horizontal offset in pixels (positional arg 0). |
| `y` | `int` | `0` | Top vertical offset in pixels (positional arg 1). |
| `width` / `w` | `int` | *Remaining* | Width of cropped region in pixels (positional arg 2). |
| `height` / `h` | `int` | *Remaining* | Height of cropped region in pixels (positional arg 3). |

## Behavior
- Automatically clamps crop boundaries within source tensor bounds to prevent out-of-bounds indexing.
- Slices tensor axes directly and packs contiguous sub-region buffers.
- Preserves layout (`HWC`/`CHW`), color space, and data type without pixel re-quantization.
