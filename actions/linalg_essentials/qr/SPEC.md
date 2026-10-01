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

## Component shapes
For input `[M,N]`, output is `Composite([Q, R])`, where Q has shape `[M,N]` and R has shape `[N,N]`. The optional `get_output_components` export describes both components for static checking and runtime validation.

Tap the result into `$parts`, then use `$parts[0]` for Q and `$parts[1]` for R. Chained tensor indexing, such as `$parts[0][0]`, selects a row of Q.

## Ordered shape contract
The optional `get_output_value_shape` export accepts and returns recursive `ValueShape` descriptors. Composite components are accessed by their original zero-based positions. The contract validates component types, arity, dimensional constraints, and output sizes; execution verifies the produced payload against the predicted output tree. See the project shape-contract documentation for the dimensional rules.
