# `dot`

Computes the inner product between two 1D vectors ($u \cdot v = \sum u_i v_i$).

## Interface
- **Input Type**: `DataType::Composite` (`Payload::Composite([u, v])`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
None.

## Behavior
- Computes dot product in parallel using Rayon reduction.

## Ordered shape contract
The optional `get_output_value_shape` export accepts and returns recursive `ValueShape` descriptors. Composite components are accessed by their original zero-based positions. The contract validates component types, arity, dimensional constraints, and output sizes; execution verifies the produced payload against the predicted output tree. See the project shape-contract documentation for the dimensional rules.
