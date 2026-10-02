# `repeat`

Repeats/tiles an input multi-dimensional tensor along specified dimensions.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `repeats` | `string` | *Required* | Repeat counts per dimension, e.g. `"[2, 3]"` or `"2, 3"`. (Positional arg 0). |

## Behavior
- Allocates a new tensor with repeated element tiles along each axis.

## OpenBLAS backend
This action belongs to `tensor_blas/0.1.0`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.
