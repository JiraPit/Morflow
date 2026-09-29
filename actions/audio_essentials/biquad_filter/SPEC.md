# `biquad_filter`

Direct Form II Transposed IIR digital biquad filter for multi-channel audio equalization and frequency filtering.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Supported DTypes**: `F32`
- **Supported Layouts**: `Planar [Channels, Samples]`, `1D [Samples]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `type` | `string` | `"lowpass"` | Filter topology: `"lowpass"`, `"highpass"`, `"bandpass"`, `"notch"`, `"peaking"`. |
| `freq` | `float` | `1000.0` | Cutoff or center frequency in Hertz ($f > 0$). |
| `q` | `float` | `0.707` | Quality factor ($Q > 0$). |
| `gain_db` | `float` | `0.0` | Boost/cut gain in dB (used for peaking filter). |
| `sample_rate` | `float` | `44100.0` | Sampling rate in Hz (automatically read from `Payload::Audio` if present). |

## Behavior
- Computes standard Robert Bristow-Johnson (RBJ) Audio EQ Cookbook coefficients.
- Processes each audio channel independently in parallel across Rayon worker threads.
- Operates in-place on sample buffers without heap reallocations.
