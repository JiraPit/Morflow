# `var`

Computes the sample or population variance $\sigma^2$ along a specified axis or across all elements.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | *None* | Axis along which variance is computed. (Positional arg 0). |
| `unbiased` | `bool` | `true` | Whether to use Bessel's correction ($N - 1$). |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Two-pass parallel variance computation using Rayon.
