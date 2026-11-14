# `stft`

Short-Time Fourier Transform computing magnitude spectrograms via Radix-2 Cooley-Tukey FFT.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Tensor`
- **Supported DTypes**: `F32`
- **Supported Layouts**: Input: `Planar [Channels, Samples]`, Output: `[Channels, FreqBins, TimeFrames]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_fft` | `int` | `1024` | FFT analysis window size in samples (positional arg 0). |
| `hop_size` / `hop_length` | `int` | `256` | Hop length in samples between successive STFT frames. |

## Behavior
- Windows overlapping `n_fft`-sample frames using a symmetric Hann window: $w[n] = 0.5(1 - \cos(2\pi n / (n_{fft} - 1)))$. Frames start at multiples of `hop_size` without padding.
- Computes single-sided $(N/2 + 1)$ magnitude spectrum bins using in-place Radix-2 Cooley-Tukey FFT, where $N$ is the smallest power of two at least $\max(n_{fft}, 16)$; each frame is zero-padded to $N$.
- Analysis frames computed concurrently across Rayon worker threads.
