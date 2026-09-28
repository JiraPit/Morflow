# `rotate`

2D affine image rotation with bilinear interpolation and canvas expansion.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
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
- Rows rendered in parallel across Rayon worker threads.
