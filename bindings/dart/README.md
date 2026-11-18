# Morflow for Dart

Run declarative `.morf` data processing pipelines over tensors, images and audio from Dart and Flutter, with the native Morflow engine underneath. The API matches the Node.js package method for method.

Supported platforms: Linux (x86_64, aarch64), macOS (Apple Silicon) and Windows (x86_64), including Flutter desktop apps, and Flutter apps on Android (arm64, x86_64) and iOS (devices and the simulator).

## Install

```sh
dart pub add morflow
```

Pipelines run published actions, which the Morflow CLI downloads ahead of time. Install the CLI with the installer from [morflow.org](https://morflow.org), then prepare each pipeline:

```sh
morflow prep pipeline.morf
```

### Android and iOS

Apps cannot download actions while they run, so a Flutter app for Android or iOS bundles whole action packs instead. List them, at exact versions, in the app's `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    morflow:
      packs:
        base: 0.3.4
        audio_basics: 0.2.4
```

`flutter build` downloads each pack for the app's platform from its GitHub release, checks it against the release's checksums and bundles it. A pipeline can use any action in a bundled pack, at `latest` or at the bundled version. The CLI is not needed for mobile apps.

Packs that need a plugin (`image_opencv`, `linalg_blas`, `tensor_blas`) cannot be bundled; use their plugin-free counterparts. The `packs` list is ignored for desktop builds, which use the action cache.

## Use

```dart
import 'dart:typed_data';
import 'package:morflow/morflow.dart' as morflow;

Future<void> main() async {
  final pipeline = morflow.fromStr(r'''
    accept Tensor $tensor
    $tensor >> base/latest/identity >> emit
  ''');

  final output = await pipeline.run<morflow.MorflowTensor>(Float32List.fromList([1, 2, 3]));
  print(output.shape);            // [3]
  print(output.toFloat32Array()); // [1.0, 2.0, 3.0]

  pipeline.close();
}
```

`morflow.load(path)` compiles a `.morf` file instead.

### Running

| Method | Returns |
| --- | --- |
| `run(input)` / `runSync(input)` | The single emitted output |
| `runAll(input)` / `runSyncAll(input)` | Every named output, as a `Map` |
| `runArgs([...])` / `runSyncArgs([...])` | The single output, binding parameters in declaration order |
| `runAllArgs([...])` / `runSyncAllArgs([...])` | Every named output, binding parameters in declaration order |

The asynchronous methods run on a native worker thread, so they never block your isolate. An output is a `MorflowTensor`, or a `List` for a Composite; name the type you expect, as in `run<MorflowTensor>(input)`.

### Inputs

| Dart value | Sent as |
| --- | --- |
| `Float32List`, `Float64List`, `Int8List`, `Int16List`, `Int32List`, `Int64List`, `Uint32List`, `Uint64List` | A 1-D tensor of that dtype |
| `Uint8List` | Raw bytes |
| `TensorInput(data:, shape:, dtype:, payloadType:, …)` | A tensor with a shape, or an `'image'` / `'audio'` payload |
| `List` of the above | A Composite |
| `num`, `String`, `bool` (in `…Args` methods) | Scalar and `*Arg` parameters |

There is no type inference: a rank-3 array is a tensor unless `payloadType: 'image'` says otherwise.

### Outputs

A `MorflowTensor` has `shape`, `dtype`, `rank`, `length` and the raw `data`, with typed views: `toTypedArray()`, `toFloat32Array()`, `toFloat64Array()`, `toInt32Array()`, `toInt64Array()`, `toUint64Array()` and so on.

### Errors and resources

Engine errors throw `MorflowException`; unsupported Dart values throw `ArgumentError`. A pipeline holds native resources, released by `close()` or automatically once it is garbage collected.

## Native library

The package's build hook downloads the native library for the target platform from the engine's GitHub release for this package version, checks it and caches it, so you never load it yourself. Other sources, set in your app's `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    morflow:
      # A library you built from bindings/capi.
      native_lib: path/to/libmorflow_capi.so
      # Or a mirror of the releases (a URL or a directory) with the same
      # <tag>/<file> layout as https://github.com/JiraPit/Morflow/releases/download.
      release_url: https://mirror.example.com/morflow
```

## Development

Inside a Morflow checkout, the build hook compiles the native library with `cargo`. Tests that execute actions need an action cache built from the checkout:

```sh
bash scripts/build-actions.sh
python3 scripts/prepare_test_actions.py --profile release --cache /tmp/morflow-actions
cd bindings/dart && MORFLOW_ACTIONS_PATH=/tmp/morflow-actions dart test
```

The Flutter app in `example/` runs every bundleable pack on a device. To test it against libraries built from the checkout, stage them like releases and add `release_url: ../../../target/staged-releases` to its `hooks.user_defines.morflow`:

```sh
python3 scripts/stage_release.py --platform android-x86_64 --out target/staged-releases
cd bindings/dart/example && flutter test integration_test
```
