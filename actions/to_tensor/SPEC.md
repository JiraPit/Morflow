# `to_tensor`

Standardizes image and audio payloads into unified multi-dimensional tensor representations with configurable layout, data type, and normalization.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: Input: `U8`, `F32`; Output: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `Planar [Channels, Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `color` / `color_space` | `string` | *None* | Target color space: `"rgb"`, `"rgba"`, `"bgr"`, `"bgra"`, `"grayscale"` (positional arg 0). |
| `dtype` | `string` | `"f32"` | Target tensor data type: `"f32"` or `"u8"`. |
| `layout` | `string` | `"hwc"` | Target spatial memory layout: `"hwc"` or `"chw"`. |
| `normalize` | `bool` | *Contextual* | Normalizes values ($[0, 255] \rightarrow [0.0, 1.0]$); defaults to `true` when converting `U8` $\rightarrow$ `F32`. |

## Behavior
- Performs color space conversions, channel re-ordering, layout transposition, and type casting in parallel using Rayon.
- Unwraps `Payload::Audio` directly into its underlying multi-channel tensor.
