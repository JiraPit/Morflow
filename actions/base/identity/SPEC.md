# `identity`

Zero-overhead pass-through action for pipeline routing, benchmarking, and debugging.

## Interface
- **Input Type**: `DataType::Bytes` (`Payload::*`)
- **Output Type**: `DataType::Bytes` (`Payload::*`)
- **Supported DTypes**: `F32`, `F64`, `U8`, `I8`, `I16`, `I32`, `I64`, `U32`, `U64`
- **Supported Layouts**: Any

## Parameters
*None*

## Behavior
- Strips wrapping arguments (`Payload::WithArgs`) and forwards the underlying payload unmodified.
- Executes with zero memory copies and zero heap reallocations.
