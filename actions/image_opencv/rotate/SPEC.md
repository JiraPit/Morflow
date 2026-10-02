# `rotate`

2D affine image rotation with bilinear interpolation and canvas expansion.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `angle` / `angle_deg` | `float` | `90.0` | Clockwise rotation angle in degrees (positional arg 0). |
| `expand` / `expand_canvas` | `bool` | `true` | Expands canvas bounds to contain full rotated image. |
| `fill` / `fill_value` | `float` | `0.0` | Background fill value for unmapped canvas regions. |

## Behavior
- Computes 2D inverse rotation mapping around image center coordinates.
- Samples source coordinates using bilinear interpolation; assigns `fill` to unmapped canvas regions.
- Uses shared OpenCV rotate for quarter turns and warpAffine otherwise. Interpolation and boundary pixels can differ from basics; CHW input retains CHW output.

## Installation
See `backends/opencv/README.md`. OpenCV is loaded only during execution.
