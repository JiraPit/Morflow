# `gain`

Multi-channel audio amplitude scaling and gain adjustment in linear factor or decibels.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Audio` / `Payload::Tensor` / `Payload::Data`)
- **Output Type**: `DataType::Tensor` (`Payload::Audio` / `Payload::Tensor` / `Payload::Data`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `linear` | `float` | `1.0` | Linear amplitude scaling factor ($S > 0$, positional arg 0). |
| `db` | `float` | *None* | Gain adjustment in decibels ($S = 10^{\text{db}/20}$). |

## Behavior
- Multiplies input audio samples directly by the resolved linear multiplier: $y[n] = x[n] \times S$.
- Executes sample-level multiplication across Rayon worker threads in parallel.
- Operates in-place on sample buffers without additional heap allocations.
