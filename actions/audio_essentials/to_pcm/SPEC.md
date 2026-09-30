# `to_pcm`

Converts structured `Audio` or multidimensional sample `Tensor` into a raw PCM byte buffer (`Payload::Data`).

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Bytes` (`Payload::Data`)
- **Supported DTypes**: `F32` (normalized `[-1.0, 1.0]`)
- **Supported Layouts**: `Planar [Channels, Samples]` or `Interleaved [Samples, Channels]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` / `format` | `string` | `"i16"` | Target integer or float PCM format: `"i16"`, `"f32"`, `"i24"`, `"i32"`, `"u8"`. |
| `layout` | `string` | `"interleaved"` | Output memory buffer ordering: `"interleaved"` (packed frames) or `"planar"`. |
| `clip` | `bool` | `true` | Clamp float samples to `[-1.0, 1.0]` before quantization to prevent integer overflow. |

## Behavior
- Quantizes normalized `f32` samples to the requested integer PCM format (`i16`, `i24`, `i32`, `u8`) or copies as IEEE `f32`.
- Handles interleaving for multichannel streams so that the output buffer is immediately compatible with hardware audio devices (ALSA, WASAPI, CoreAudio) or socket streaming.
- Fully parallelized channel interleaving and quantization with Rayon.
