# `norm`

Computes the matrix or vector $p$-norm ($L_1$, $L_2$ Euclidean, or $L_\infty$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `p` | `float` / `string` | `2.0` | Order of the norm (`1.0`, `2.0`, `"inf"`). (Positional arg 0). |
| `axis` / `dim` | `int` | *None* | Axis along which norm is computed. |
| `keepdim` | `bool` | `false` | Whether to retain reduced dimension with size 1. |

## Behavior
- Parallel norm calculation across CPU threads using Rayon.
