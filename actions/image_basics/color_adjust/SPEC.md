# `color_adjust`

Color grading and photographic tone adjustments (brightness, contrast, gamma, saturation, exposure).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `brightness` | `float` | `0.0` | Additive brightness offset ($[-1.0, 1.0]$ or $[-255, 255]$). |
| `contrast` | `float` | `1.0` | Multiplicative contrast scaling around midpoint 0.5. |
| `gamma` | `float` | `1.0` | Non-linear power-law gamma correction ($\gamma > 0$). |
| `saturation` / `sat` | `float` | `1.0` | Rec.601 luma-weighted color saturation multiplier. |
| `exposure` | `float` | `0.0` | Exposure adjustment in EV stops ($2^{\text{exposure}}$). |

## Behavior
- Executes photographic color adjustment pipeline: exposure $\rightarrow$ contrast $\rightarrow$ brightness $\rightarrow$ gamma $\rightarrow$ saturation.
- Uses ITU-R BT.601 luma coefficients ($Y = 0.299R + 0.587G + 0.114B$) for RGB saturation interpolation.
- Parallelized across pixel rows/chunks via Rayon; clamps output values to standard dynamic ranges.
