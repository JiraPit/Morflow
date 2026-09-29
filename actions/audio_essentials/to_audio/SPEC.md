# `to_audio`

Converts binary audio file data (e.g. WAV format), raw PCM sample buffers, or multidimensional numeric tensors into a structured, zero-copy `Payload::Audio` stream.

## Interface
- **Input Type**: `DataType::RawBytes` / `DataType::Tensor` (`Payload::Data` or `Payload::Tensor`)
- **Output Type**: `DataType::Audio` (`Payload::Audio`)
- **Supported DTypes**: `F32` (normalized `[-1.0, 1.0]`)
- **Supported Layouts**: Standard `Planar [Channels, Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sample_rate` | `int` | `44100` (or inferred from WAV header) | Sampling rate in Hz. Auto-detected when input is WAV. |
| `channels` | `int` | `2` (or inferred from WAV / tensor shape) | Number of audio channels. Auto-detected when input is WAV. |
| `dtype` / `format` | `string` | `"i16"` | Sample encoding format for raw PCM bytes: `"i16"`, `"f32"`, `"i24"`, `"i32"`, `"u8"`. |
| `layout` | `string` | `"interleaved"` | Memory ordering for raw PCM bytes: `"interleaved"` or `"planar"`. |
| `normalize` | `bool` | `true` | Scale integer PCM samples to `[-1.0, 1.0]` floating-point range. |

## Behavior
- **RIFF / WAV Automatic Ingestion**: When input `Payload::Data` contains a valid RIFF/WAVE header, automatically parses chunk headers, sampling rate, channels, bit depth, and decodes interleaved PCM or IEEE float samples into planar float32 channels.
- **Raw PCM Ingestion**: Decodes raw byte buffers according to `dtype`, `channels`, and `layout` arguments into planar float32.
- **Tensor Ingestion**: Standardizes 1D `[samples]` (mono) or 2D `[channels, samples]` tensors into `Payload::Audio` with proper channel layout metadata.
- **Audio Pass-Through**: If already `Payload::Audio`, updates sample rate or layout if explicitly passed as arguments.
- High-throughput parallel de-interleaving and format conversion using Rayon work-stealing.
