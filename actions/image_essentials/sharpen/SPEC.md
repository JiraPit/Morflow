# `sharpen`

Unsharp masking filter to enhance high-frequency spatial edge details.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strength` / `amount` | `float` | `1.0` | High-frequency sharpening factor ($S \ge 0$, positional arg 0). |
| `sigma` | `float` | `1.0` | Standard deviation of Gaussian blur used for unsharp mask ($\sigma > 0$). |
| `radius` | `int` | $\lceil 3\sigma \rceil$ | Gaussian blur kernel radius in pixels. |

## Behavior
- Computes low-pass base via Gaussian blur: $I_{\text{blur}} = \text{GaussianBlur}(I, \sigma)$.
- Adds high-frequency detail mask back to input: $I_{\text{sharp}} = I + \text{strength} \times (I - I_{\text{blur}})$.
- Parallelized per row using Rayon; clamps output pixel values to valid dynamic ranges.
