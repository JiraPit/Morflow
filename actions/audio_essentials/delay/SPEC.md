# `delay`

Audio delay feedback line with adjustable wet/dry mix and feedback loop attenuation.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Audio` (`Payload::Audio`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `time_ms` / `time` | `float` | `120.0` | Delay time in milliseconds ($t > 0$, positional arg 0). |
| `feedback` | `float` | `0.35` | Feedback loop recirculation gain ($0.0 \le \text{feedback} < 1.0$). |
| `mix` / `wet` | `float` | `0.3` | Wet/dry blend factor ($0.0 = \text{100\% dry}$, $1.0 = \text{100\% wet}$). |
| `sample_rate` | `float` | `44100.0` | Sampling rate in Hz (automatically read from `Payload::Audio` if present). |

## Behavior
- Allocates a circular ring buffer sized to the discrete delay sample length.
- Blends delayed signal into feedback loop and sums dry/wet components: $y[n] = (1 - \text{mix}) x[n] + \text{mix} \cdot x[n - L]$.
- Audio channels are processed independently in parallel across Rayon worker threads.
