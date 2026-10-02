# `dot`

Computes the inner product between two 1D vectors ($u \cdot v = \sum u_i v_i$).

## Interface
- **Input Type**: `DataType::Composite` (`Payload::Composite([u, v])`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: F32 computation and output; other numerical inputs are converted to F32

## Parameters
None.

## Behavior
- Computes the dot product using OpenBLAS `cblas_sdot`.

## Ordered shape contract
`shapecheck` validates the ordered Composite inputs and predicts the output type and dimensions. Component indexes retain their original order. The engine verifies the returned payload against that prediction during execution.

## OpenBLAS backend
This action belongs to `linalg_blas/0.1.2`. It uses the host's shared LP64 OpenBLAS library during execution. OpenBLAS is not bundled in the action binary. Shapechecking and pipeline loading do not load OpenBLAS. Set `MORFLOW_OPENBLAS_LIBRARY` to select an explicit shared library path. There is no fallback to the basics implementation if the library is missing.

## Runtime plugin

Declare `plugin openblas/0.1.1` in the pipeline and run `morflow prep`. These actions require the `openblas` plugin with a compatible `^0.1.0` version. The plugin loads shared system OpenBLAS during execution; shape checking remains offline. See `plugins/openblas/README.md` for system requirements.
