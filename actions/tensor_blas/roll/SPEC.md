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
This action belongs to `tensor_blas/0.1.2`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.

## Runtime plugin

Declare `plugin openblas/0.1.1` in the pipeline and run `morflow prep`. These actions require the `openblas` plugin with a compatible `^0.1.0` version. The plugin loads shared system OpenBLAS during execution; shape checking remains offline. See `plugins/openblas/README.md` for system requirements.
