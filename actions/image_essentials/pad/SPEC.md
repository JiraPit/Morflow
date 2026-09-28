# `pad`

Spatial border padding and canvas extension with multiple boundary modes.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pad` | `int` | *None* | Uniform padding applied to all four borders (positional arg 0). |
| `top` / `pad_top` | `int` | `0` | Top border padding in pixels. |
| `bottom` / `pad_bottom` | `int` | `0` | Bottom border padding in pixels. |
| `left` / `pad_left` | `int` | `0` | Left border padding in pixels. |
| `right` / `pad_right` | `int` | `0` | Right border padding in pixels. |
| `mode` / `pad_mode` | `string` | `"constant"` | Boundary mode: `"constant"`, `"edge"` (`"replicate"`, `"clamp"`), `"reflect"` (`"mirror"`). |
| `fill` / `fill_value` | `float` | `0.0` | Constant fill value when `mode` is `"constant"`. |

## Behavior
- Allocates expanded destination buffer with dimensions `[H + top + bottom, W + left + right, C]`.
- Maps out-of-boundary pixel indices according to selected mode (`constant`, clamped `edge`, or mirrored `reflect`).
- Rows rendered in parallel across Rayon worker threads.
