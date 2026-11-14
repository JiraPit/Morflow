# `noise_gate`

Downward audio expander and noise gate with hold timer and attack/release envelopes.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Audio` (`Payload::Audio`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `threshold_db` / `threshold` | `float` | `-40.0` | Gate opening threshold in dBFS. |
| `ratio` | `float` | `10.0` | Downward expansion attenuation ratio ($R \ge 1.0$). |
| `attack_ms` / `attack` | `float` | `2.0` | Gate opening attack time in milliseconds. |
| `hold_ms` / `hold` | `float` | `10.0` | Hold time in milliseconds before release begins. |
| `release_ms` / `release` | `float` | `50.0` | Gate closing release time in milliseconds. |
| `sample_rate` / `rate` | `float` | *Audio sample rate* | Sampling rate in Hz; overrides the rate read from the input `Payload::Audio`. |

## Behavior
- Attenuates signals falling below `threshold_db` via downward expansion.
- Maintains open state for `hold_ms` after level drops below threshold to prevent gating chatter.
- Audio channels are processed independently in parallel across Rayon worker threads.
