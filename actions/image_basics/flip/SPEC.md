# `flip`

Spatial tensor mirroring along horizontal and vertical axes.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `U8`, `I32`
- **Supported Layouts**: `HWC`, `CHW`, `2D [H, W]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` | `string` | *Required* | Mirror axis: `"horizontal"` (`"h"`, `"x"`, `"1"`), `"vertical"` (`"v"`, `"y"`, `"0"`), `"both"` (`"hv"`, `"xy"`, positional arg 0). |

## Behavior
- Reverses spatial indexing order along the target dimension(s).
- Parallelized across rows or channels using Rayon.
- Returns unmodified tensor payload if no valid flip axis is provided.
