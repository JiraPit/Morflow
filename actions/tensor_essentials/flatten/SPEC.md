# `flatten`

Flattens a contiguous range of dimensions into a single dimension.

## Interface
- **Input Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Output Type**: `DataType::Tensor` (`Payload::Tensor`)
- **Supported DTypes**: All

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `start_dim` | `int` | `0` | Starting dimension index to flatten. (Positional arg 0). |
| `end_dim` | `int` | `-1` | Ending dimension index to flatten (inclusive; `-1` indicates final dimension). (Positional arg 1). |

## Behavior
- Collapses specified dimension span into a single product dimension.
- Contiguous views flatten with **0 memory allocations**.
