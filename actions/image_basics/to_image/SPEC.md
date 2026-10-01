# `to_image`

Converts raw tensors into strongly-typed `Image` payloads with format validation and denormalization.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Image` (`Payload::Image`)
- **Supported DTypes**: Input: `F32`, `U8`; Output: `U8`, `F32`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `color` / `color_space` | `string` | *None* | Target color space: `"rgb"`, `"rgba"`, `"bgr"`, `"bgra"`, `"grayscale"` (positional arg 0). |
| `dtype` | `string` | `"u8"` | Target pixel data type: `"u8"` or `"f32"`. |
| `layout` | `string` | `"hwc"` | Target image memory layout: `"hwc"` or `"chw"`. |
| `denormalize` / `unnormalize` | `bool` | `true` | Scales normalized $[0.0, 1.0]$ floats to $[0, 255]$ unsigned 8-bit integers. |

## Behavior
- Validates 2D/3D tensor shape integrity and channel counts for valid image representation.
- Scales and converts float sample ranges to byte buffers when `denormalize=true`.
- Constructs a validated `core_types::Image` payload encapsulating tensor and metadata.
