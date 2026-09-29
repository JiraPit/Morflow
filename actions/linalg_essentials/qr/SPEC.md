# `qr`

Computes the QR decomposition of a 2D matrix ($A = Q R$), returning composite payload `[Q, R]`.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Composite` (`Payload::Composite([Q, R])`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes orthogonal matrix $Q$ and upper triangular matrix $R$ using Modified Gram-Schmidt orthogonalization.
