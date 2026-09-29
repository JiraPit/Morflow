# `std`

Computes the sample or population standard deviation $\sigma$ along a specified axis or across all elements.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | *None* | Axis along which standard deviation is computed. (Positional arg 0). |
| `unbiased` | `bool` | `true` | Whether to use Bessel's correction ($N - 1$). |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Parallel standard deviation computation using Rayon.
