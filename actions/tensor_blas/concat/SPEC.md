# `concat`

Concatenates multiple tensors along a specified axis.

## Interface
- **Input Type**: `DataType::Composite` or `DataType::Tensor` (`Payload::Composite([Tensor, ...])` / `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `axis` / `dim` | `int` | `0` | Axis along which tensors are concatenated. (Positional arg 0). |

## Behavior
- Verifies rank and dimensional alignment across non-concatenated axes.
- Performs parallel block copying into the resulting combined contiguous tensor buffer.

## Ordered shape contract
`shapecheck` validates the ordered Composite inputs and predicts the output type and dimensions. Component indexes retain their original order. The engine verifies the returned payload against that prediction during execution.

## OpenBLAS backend
This action belongs to `tensor_blas/0.1.0`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.
