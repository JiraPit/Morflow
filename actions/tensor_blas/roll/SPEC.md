# `roll`

Circularly shifts elements of a tensor along a specified axis.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shift` / `shifts` | `int` | `0` | Number of positions elements are shifted. (Positional arg 0). |
| `axis` / `dim` | `int` | `0` | Axis along which elements are shifted. (Positional arg 1). |

## Behavior
- Slices tensor along the split boundary and concatenates parts in reverse order.

## OpenBLAS backend
This action belongs to `tensor_blas/0.1.0`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.
