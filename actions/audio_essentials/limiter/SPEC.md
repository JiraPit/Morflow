# `limiter`

Fast peak limiter and soft-clipping saturation for peak ceiling enforcement and loudness maximization.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Audio` (`Payload::Audio`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ceiling_db` / `ceiling` | `float` | `-0.1` | Peak output ceiling in dBFS ($\le 0.0$, positional arg 0). |
| `release_ms` / `release` | `float` | `50.0` | Peak envelope recovery release time in milliseconds. |
| `mode` | `string` | `"brickwall"` | Limiting algorithm: `"brickwall"` or `"soft_clip"` (`"clip"`, `"saturate"`). |
| `drive` | `float` | `1.0` | Saturation drive multiplier for soft-clipping mode. |
| `sample_rate` | `float` | `44100.0` | Sampling rate in Hz (automatically read from `Payload::Audio` if present). |

## Behavior
- In `"brickwall"` mode, tracks peak envelope with exponential release decay and clamps peaks strictly at the ceiling.
- In `"soft_clip"` mode, applies hyperbolic tangent non-linear saturation: $y = C \cdot \tanh((x \cdot D) / C)$.
- Channels are processed concurrently across Rayon worker threads.
