# `gaussian_blur`

2D spatial Gaussian smoothing and box filtering for noise reduction and image softening.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sigma` | `float` | `1.0` | Standard deviation of the Gaussian kernel ($\sigma > 0$, positional arg 0). |
| `radius` | `int` | $\lceil 3\sigma \rceil$ | Kernel radius in pixels. |
| `mode` / `type` | `string` | `"gaussian"` | Smoothing kernel type: `"gaussian"` or `"box"`. |

## Behavior
- Implements separable 1D horizontal and vertical convolution passes for $O(K \times W \times H)$ efficiency.
- Normalizes kernel weights dynamically: $G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2 / (2\sigma^2)}$.
- Replicates edge boundary pixels; parallelized per row across Rayon worker threads.
