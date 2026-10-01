# `to_tensor`

Standardizes image and audio payloads into unified multi-dimensional tensor representations with configurable layout, data type, and normalization.

## Interface
- **Input Type**: `DataType::Tensor` / `DataType::Image` / `DataType::Audio` (`Payload::Tensor` / `Payload::Image` / `Payload::Audio`)
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
- This is the designated bridge from `Image` and `Audio` payloads into plain `Tensor` payloads; the reverse conversions live in `image_basics/to_image` and `audio_basics/to_audio`.
- Input and output types are both strictly enforced at runtime, so this action is the only valid way to feed image or audio data into tensor-native actions.
- Performs color space conversions, channel re-ordering, layout transposition, and type casting in parallel using Rayon.
- Unwraps `Payload::Audio` directly into its underlying multi-channel tensor.
