# `stereo_widen`

Mid/Side matrix audio stereo field widener and center channel balance control.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Audio` / `Payload::Tensor`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [2, Samples]`, `Planar [Channels >= 2, Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` / `amount` | `float` | `1.2` | Stereo width factor ($0.0 = \text{mono}$, $1.0 = \text{unchanged}$, $>1.0 = \text{widened}$). |
| `center_gain_db` / `center` | `float` | `0.0` | Mid/center channel boost or cut in dB. |

## Behavior
- Decodes Left/Right stereo channels into Mid/Side components: $M = (L + R)/\sqrt{2}$, $S = (L - R)/\sqrt{2}$.
- Applies center gain and side width factors: $M' = M \times 10^{\text{gain}/20}$, $S' = S \times \text{width}$.
- Re-encodes back into discrete stereo: $L' = (M' + S')/\sqrt{2}$, $R' = (M' - S')/\sqrt{2}$.
- Sample buffers processed in parallel using Rayon.
