# `normalize`

Audio peak and RMS level normalization to target amplitude or decibel headroom.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `string` | `"peak"` | Normalization metric: `"peak"` or `"rms"`. |
| `target_peak` | `float` | `1.0` | Target linear amplitude level ($P > 0$). |
| `target_peak_db` / `target_db` | `float` | *None* | Target level in decibels ($P = 10^{\text{db}/20}$). |

## Behavior
- Measures global peak absolute value ($\max |x[n]|$) or root-mean-square ($\sqrt{\frac{1}{N}\sum x[n]^2}$) across all channels.
- Scales all channels uniformly by $S = P_{\text{target}} / P_{\text{current}}$ to preserve multichannel balance.
- Parallelized across Rayon worker threads; bypasses scaling on silent input buffers.
