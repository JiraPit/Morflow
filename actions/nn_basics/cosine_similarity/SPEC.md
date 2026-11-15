# `cosine_similarity`

Computes the cosine similarity between two tensors along a specified axis: $\frac{A \cdot B}{\|A\|_2 \|B\|_2}$.

## Interface
- **Input Type**: `DataType::Composite | DataType::Tensor` (`Payload::Composite([Tensor, Tensor])`, or a single `Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`

## DTypes
- The two tensors are combined like the two-tensor arithmetic actions: the same dtype stays the same; with a float involved the result is `F64` if either side is `F64`, otherwise `F32`; two integers give the wider (or smallest common signed) integer type.
- An `F32` result is computed in single precision exactly as before, `F64` in double precision, and an integer result is turned into `F64`.
- A single Tensor input is passed through, with integer tensors converted to `F64`.
- NaN values in the input behave like NumPy (they propagate through the result).

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `eps` | `float` | `1e-8` | Small epsilon constant to prevent division by zero. |

## Behavior
- Computes normalized dot products between paired vectors in parallel using Rayon.

## Ordered shape contract
The optional `get_output_value_shape` export accepts and returns recursive `ValueShape` descriptors. Composite components are accessed by their original zero-based positions. The contract validates component types, arity, dimensional constraints, and output sizes; execution verifies the produced payload against the predicted output tree. See the project shape-contract documentation for the dimensional rules.
