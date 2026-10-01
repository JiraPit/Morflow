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
The optional `get_output_value_shape` export accepts and returns recursive `ValueShape` descriptors. Composite components are accessed by their original zero-based positions. The contract validates component types, arity, dimensional constraints, and output sizes; execution verifies the produced payload against the predicted output tree. See the project shape-contract documentation for the dimensional rules.
