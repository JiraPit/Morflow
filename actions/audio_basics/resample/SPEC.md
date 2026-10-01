# `resample`

Band-limited windowed sinc interpolation for high-quality audio sample rate conversion.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Audio` (`Payload::Audio`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `from_rate` / `source_rate` | `float` | `Audio.sample_rate` | Source sample rate in Hz (automatically read from `Payload::Audio`). |
| `to_rate` / `rate` | `float` | `44100.0` | Target sample rate in Hz (positional arg 0). |

## Behavior
- Performs band-limited sinc interpolation with Blackman-Harris windowing (kernel half-length 8).
- Scales cutoff frequency to $0.95 \times \min(1.0, \text{to\_rate}/\text{from\_rate})$ to eliminate downsampling aliasing.
- Multi-channel audio streams are resampled in parallel across Rayon worker threads.

## Shape Contract
- With a known `from_rate` and target rate, predicts the output sample count using the same rounding as execution and preserves the planar channel count. One-channel planar input produces a one-dimensional mono output.
- Without `from_rate`, the source rate comes from audio metadata at execution, so the static output shape is unknown. Unresolved rate arguments also produce an unknown verdict.
- Rejects malformed, non-positive, or non-finite rates. Execution accepts mono and planar audio layouts.
