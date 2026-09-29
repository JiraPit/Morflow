# `resize`

Spatial image resampling using interpolation algorithms (Nearest Neighbor, Bilinear, Bicubic, Area Box).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` / `w` | `int` | *None* | Target width in pixels (positional arg 0). |
| `height` / `h` | `int` | *None* | Target height in pixels (positional arg 1). |
| `scale` | `float` | *None* | Uniform scaling factor for width and height. |
| `scale_x` | `float` | *None* | Horizontal scaling factor. |
| `scale_y` | `float` | *None* | Vertical scaling factor. |
| `filter` | `string` | `"bilinear"` | Resampling kernel: `"nearest"`, `"bilinear"`, `"bicubic"`, `"area"`. |
| `keep_aspect_ratio` | `bool` | `false` | Preserves original aspect ratio when fitting into target dimensions. |

## Behavior
- Automatically computes target dimensions from explicit dimensions, scaling factors, or aspect ratio constraints.
- Processes rows in parallel across Rayon worker threads.
- Clamps pixel values to valid dynamic ranges (`[0.0, 1.0]` for `F32`, `[0, 255]` for `U8`).
