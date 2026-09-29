# `resample`

Band-limited windowed sinc interpolation for high-quality audio sample rate conversion.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `from_rate` / `source_rate` | `float` | `48000.0` | Source sample rate in Hz (automatically read from `Payload::Audio`). |
| `to_rate` / `rate` | `float` | `44100.0` | Target sample rate in Hz (positional arg 0). |

## Behavior
- Performs band-limited sinc interpolation with Blackman-Harris windowing (kernel half-length 8).
- Scales cutoff frequency to $0.95 \times \min(1.0, \text{to\_rate}/\text{from\_rate})$ to eliminate downsampling aliasing.
- Multi-channel audio streams are resampled in parallel across Rayon worker threads.
