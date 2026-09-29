# `stft`

Short-Time Fourier Transform computing magnitude spectrograms via Radix-2 Cooley-Tukey FFT.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor`
- **Supported DTypes**: `F32`
- **Supported Layouts**: Input: `Planar [Channels, Samples]`, Output: `[Channels, FreqBins, TimeFrames]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_fft` | `int` | `1024` | FFT analysis window size in samples (positional arg 0). |
| `hop_size` / `hop_length` | `int` | `256` | Hop length in samples between successive STFT frames. |

## Behavior
- Windows overlapping time frames using a periodic Hann window: $w[n] = 0.5(1 - \cos(2\pi n / (N - 1)))$.
- Computes single-sided $(N/2 + 1)$ magnitude spectrum bins using in-place Radix-2 Cooley-Tukey FFT.
- Analysis frames computed concurrently across Rayon worker threads.
