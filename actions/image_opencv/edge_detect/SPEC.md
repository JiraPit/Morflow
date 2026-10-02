# `edge_detect`

Spatial gradient convolution and edge detection (Sobel, Prewitt, Laplacian).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` / `filter` | `string` | `"sobel"` | Gradient operator: `"sobel"`, `"sobel_x"`, `"sobel_y"`, `"prewitt"`, `"laplacian"` (positional arg 0). |
| `strength` / `scale` | `float` | `1.0` | Output gradient scaling multiplier. |

## Behavior
- Convolves input image with standard $3 \times 3$ directional gradient kernels.
- Computes Euclidean gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ for Sobel and Prewitt modes.
- Accelerated with shared OpenCV; returns absolute directional responses.

## OpenCV backend
Uses shared OpenCV through the versioned `opencv-bridge` plugin. Declare `plugin opencv-bridge/0.1.0` in the pipeline and run `morflow prep`. Shape checking does not load OpenCV. See `backends/opencv/README.md` for installation, layout rules and numerical differences.
