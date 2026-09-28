# `threshold`

Pixel intensity thresholding and binary binarization (Binary, Inverted, Truncate, ToZero, Otsu).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Image` / `Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `threshold` / `thresh` | `float` | `0.5` (F32) / `128` (U8) | Cutoff threshold value (positional arg 0; computed if `mode="otsu"`). |
| `max_val` / `max` | `float` | `1.0` (F32) / `255` (U8) | Maximum value assigned to pixels meeting threshold criteria. |
| `mode` | `string` | `"binary"` | Thresholding mode: `"binary"`, `"binary_inv"`, `"otsu"`, `"truncate"`, `"to_zero"`, `"to_zero_inv"`. |

## Behavior
- Automatically calculates optimal threshold via intra-class variance minimization when `mode="otsu"`.
- Evaluates non-linear point mapping rules per pixel (e.g. $I(x) > T \implies \text{max\_val}$).
- Parallelized across contiguous memory chunks using Rayon.
