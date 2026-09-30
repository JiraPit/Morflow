# Native action shape contracts

A native action reports `ShapeResult::Ok`, `Unknown`, or `Invalid`:

- `Ok` supplies the output dimensions predicted from the input dimensions and action arguments.
- `Invalid` rejects rank, axis, argument, element-count, or overflow constraints before processing.
- `Unknown` means the available information cannot determine validity or output dimensions. Dynamic arguments and missing payload metadata stay unresolved.

Native actions validate these contracts at execution as well. Known output dimensions are compared with the produced payload; a mismatch returns an action error. Zero dimensions in the static callback represent wildcard lengths.

The contracts cover scalar and tensor transformations, reductions, pooling, raw tensor image transformations, mono and planar audio, and single-matrix or batched matrix operations. Their defaults, positional arguments, aliases, layout detection, and dimensional rounding follow execution. Regression tests in `pipeline/tests/action_shapes.rs` load verified native binaries and compare predictions with actual outputs.

## Metadata and composite values

`to_tensor` accepts Tensor, Image, and Audio payloads. Dimensions alone cannot identify the source payload kind, image color space, layout, or dtype. Its static callback reports `Unknown`. `to_audio` also reports `Unknown` because decoding depends on encoded contents or source metadata. Resampling without an explicit source rate similarly depends on audio metadata.

QR validates its rank-2 matrix input and returns `Unknown` for the dimensions of its composite output. Composite-input actions require contracts capable of describing each component; the current callback accepts one shape.

`identity` preserves its input dimensions. WAV and PCM encoding validate audio ranks and produce opaque Bytes, which have no tensor shape.

A known input shape enables dimensional checking, but execution can still fail for reasons involving dtype, layout, encoded contents, or numerical values. Examples include a singular inverse and a Cholesky input that is not positive definite.

## Local validation

Build the workspace and prepare its local versioned fixtures before running tests:

```sh
cargo build --workspace
python3 scripts/prepare_test_actions.py
MORFLOW_ACTIONS_PATH="$PWD/target/debug/actions" cargo test --workspace
```

Updated action packs use new patch versions so exact imports keep referring to their original published binaries. Prepare or install the new version, or refresh `latest`, to use the corrected contracts.
