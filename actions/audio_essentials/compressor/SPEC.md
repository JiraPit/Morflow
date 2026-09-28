# `compressor`

Dynamic range compression with soft-knee smoothing, attack/release ballistics, and makeup gain.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Audio` / `Payload::Tensor`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `threshold_db` / `threshold` | `float` | `-12.0` | Compression threshold in dBFS. |
| `ratio` | `float` | `4.0` | Compression slope ratio ($R \ge 1.0$). |
| `attack_ms` / `attack` | `float` | `10.0` | Envelope detector attack time in milliseconds. |
| `release_ms` / `release` | `float` | `100.0` | Envelope detector release time in milliseconds. |
| `knee_db` / `knee` | `float` | `2.0` | Soft knee width in dB. |
| `makeup_db` / `makeup` | `float` | `0.0` | Post-compression linear makeup gain in dB. |
| `sample_rate` | `float` | `44100.0` | Sampling rate in Hz (automatically read from `Payload::Audio` if present). |

## Behavior
- Computes quadratic knee interpolation around the threshold to prevent harsh compression transitions.
- Smooths dynamic gain reduction using decoupled attack and release ballistic filters.
- Audio channels are processed independently in parallel across Rayon worker threads.
