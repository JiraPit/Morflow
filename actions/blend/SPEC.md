# `blend`

Pixel compositing and color blending with solid colors or alpha layers.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `string` | `"alpha"` | Blend mode: `"alpha"`, `"multiply"`, `"screen"`, `"overlay"`, `"add"`, `"subtract"`, `"difference"`, `"darken"`, `"lighten"` (positional arg 0). |
| `opacity` / `alpha` | `float` | `1.0` | Blend strength and layer opacity ($0.0 \le \text{opacity} \le 1.0$). |
| `color` | `string` | *None* | Comma-separated RGB color channels (e.g. `"1.0,0.5,0.2"` or `"255,128,64"`). |

## Behavior
- Computes per-pixel compositing transfer function against the target color.
- Blends result with source image according to `opacity`: $C_{\text{out}} = (1 - \alpha) C_{\text{src}} + \alpha C_{\text{blend}}$.
- Parallelized across chunks using Rayon; clamps output to standard intensity range.
