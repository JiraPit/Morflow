# `sharpen`

Unsharp masking filter to enhance high-frequency spatial edge details.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
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
- Accelerated with shared OpenCV; preserves F32 values; U8 results are saturated.

## OpenCV backend
Uses shared OpenCV through the versioned `opencv-bridge` plugin. Declare `plugin opencv-bridge/0.1.2` in the pipeline and run `morflow prep`. Shape checking does not load OpenCV. See `actions/image_opencv/README.md` for installation, layout rules and numerical differences.
