# Morflow plugins

Plugins are versioned shared binaries used by groups of actions. The engine, actions and plugins are separate build and release deliverables. External libraries such as OpenCV remain system dependencies.

## Use a plugin

Declare an exact version or write `latest` explicitly:

```perl
plugin opencv-bridge/0.1.2
from image_opencv/0.1.2 import resize

accept Tensor[480,640,3] $image
$image >> resize(320,240) >> emit
```

```sh
morflow prep pipeline.morf
morflow plugins list
morflow check pipeline.morf
```

Preparation downloads the declared plugins and imported actions, verifies published checksums, and checks the action's plugin requirements. A missing declaration or incompatible version is an error. Neither plugin declarations nor action imports infer `latest`.

`morflow plugins search opencv` searches the shipped plugin catalog and prints installable paths such as `opencv-bridge/latest`. Like action search, it works offline and accepts `-l` / `--limit` (default: 5).

`morflow plugins install opencv-bridge/0.1.2` installs a plugin independently. Use `-p` / `--path` for a custom plugin cache and `-f` / `--force` to download again. An exact version or explicit `latest` is required.

`morflow plugins clean` removes prepared plugin entries. Existing loaded pipelines retain their selected snapshots.

## Cache and versions

Plugins are installed in `~/.morflow/plugins`, or `MORFLOW_PLUGINS_PATH`. CLI overrides are `morflow prep --plugins-path PATH`, `morflow check --plugins-path PATH`, and `morflow plugins list|install|clean --path PATH`.

Each plugin has a versioned binary and a checksum receipt. For example:

```text
opencv-bridge/
  opencv-bridge_plugin-0.1.1-linux-x86_64.so
  opencv-bridge_plugin-0.1.1-linux-x86_64.so.json
  opencv-bridge_plugin-latest-linux-x86_64.so
  opencv-bridge_plugin-latest-linux-x86_64.so.json
```

Every preparation or installation of `latest` selects the highest stable published plugin release. Exact versions and `latest` coexist. Matching bytes are reused, and the receipt still updates when the selected release changes. Failed downloads or checksum verification preserve the previously verified entry.

Checks and pipeline loading verify local artifacts without loading the plugin's system dependencies. Execution opens the selected plugin when an action requests it. Each pipeline keeps its selection until reloaded, including when another preparation refreshes `latest`.

## Develop a plugin

Place its Cargo package in `plugins/<name>/`. Set `crate-type = ["cdylib"]`, a unique library name, an exact package version, and `[package.metadata.morflow] plugin = "<name>"`. Keep external dependencies dynamically linked.

```sh
bash scripts/build-plugins.sh                  # All plugins
bash scripts/build-plugins.sh opencv-bridge    # One plugin
```

The script packages exact-version binaries and receipts in `target/release/plugins`. Engine loading discovers this development directory; set `MORFLOW_PLUGINS_PATH` when using another directory. Building actions or the engine does not build plugins.

An action declares its dependencies in Cargo metadata:

```toml
[[package.metadata.morflow.plugins]]
name = "opencv-bridge"
version = "^0.1.0"
```

It also exports `get_required_plugins() -> RVec<PluginRequirement>` with the same requirements. The loader checks that the export matches the verified artifact receipt. Actions without plugin dependencies do not export this function.

The engine supplies `prepared.runtime` only during execution, after `shapecheck` returns `Ready`. Use that context to resolve a symbol from a declared plugin. Each action keeps its typed native-call wrapper inside its own implementation. The plugin documents the exact signature and ownership rules; there is no separately distributed helper crate. `shapecheck` must remain independent of runtime plugin access. Contexts own their selected plugin set, and can safely be cloned across parallel calls. Once opened, native modules remain resident for the process lifetime so background workers and returned storage cannot outlive their code. Modules are cached by immutable snapshot path; this does not change which version a pipeline selects.

Plugins publish under `plugins/<name>/v<version>`. The plugin workflow builds platform artifacts and publishes checksums; every action release includes required checksummed dependency metadata, including an empty plugin list for actions without plugins. A release is usable only with the external shared-library ABI it was built against. OpenCV release notes identify the build environment; runtime loader errors identify missing dependencies.
