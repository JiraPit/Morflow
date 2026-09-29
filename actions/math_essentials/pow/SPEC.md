# `pow`

Raises all elements of an input tensor to an exponent power ($x^p$).

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: `F32`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `exponent` / `power` | `float` | `1.0` | Power exponent. (Positional arg 0). |

## Behavior
- Parallel in-place COW power calculation using Rayon.
