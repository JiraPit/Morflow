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
- Uses OpenCV resize.
- Preserves F32 range; saturates U8 outputs. OpenCV interpolation may produce different pixels from basics.

## OpenCV backend
Uses shared OpenCV through the versioned `opencv-bridge` plugin. Declare `plugin opencv-bridge/0.1.2` in the pipeline and run `morflow prep`. Shape checking does not load OpenCV. See `actions/image_opencv/README.md` for installation, layout rules and numerical differences.
