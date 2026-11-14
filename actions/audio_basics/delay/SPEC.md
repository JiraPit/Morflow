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
| `sample_rate` / `rate` | `float` | *Audio sample rate* | Sampling rate in Hz; overrides the rate read from the input `Payload::Audio`. |

## Behavior
- Allocates a circular ring buffer sized to the discrete delay sample length.
- Blends delayed signal into feedback loop and sums dry/wet components: $w[n] = x[n] + \text{feedback} \cdot w[n - L]$ and $y[n] = (1 - \text{mix}) x[n] + \text{mix} \cdot w[n - L]$, where $L$ is the delay in samples.
- Audio channels are processed independently in parallel across Rayon worker threads.
