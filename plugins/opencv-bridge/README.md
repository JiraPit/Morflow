# OpenCV bridge plugin

This prebuilt Rust plugin connects `image_opencv` actions to shared system OpenCV. It uses the Rust `opencv` crate with only the `core` and `imgproc` bindings. OpenCV is not bundled or statically linked: it stays a dependency you install through your system package manager.

Declare `plugin opencv-bridge/0.1.3` in the pipeline and run `morflow prep`.

## OpenCV requirement

Each published binary loads one specific OpenCV version. OpenCV versions its shared libraries per minor release (`libopencv_core.so.406` is OpenCV 4.6) and does not promise compatibility between them, so a binary built against one version cannot load another.

<!-- Update this table whenever a release is built against different OpenCV versions. -->
| Platform | 0.1.3 is built against | Install |
| --- | --- | --- |
| Linux x86_64 and aarch64 | OpenCV 4.6 (Ubuntu 24.04) | `sudo apt install libopencv-core406t64 libopencv-imgproc406t64` |
| macOS, Apple Silicon | Homebrew `opencv@4` 4.14 | `brew install opencv@4` |
| Windows x86_64 | vcpkg `opencv4` 4.14, `x64-windows` triplet | `vcpkg install opencv4:x64-windows`, then add its `installed/x64-windows/bin` folder to `PATH` |

The published Linux binaries also need a glibc and C++ runtime at least as new as Ubuntu 24.04's. Ubuntu 24.04 and distributions based on it work as is.

Homebrew upgrades `opencv@4` in place. If it moves past the version in the table, the published macOS binary stops loading until a plugin release is built against the new version; build from source in the meantime.

### Other OpenCV versions: build from source

On any other OpenCV version, such as Fedora, Arch or a newer Ubuntu, running a pipeline fails with an error like:

```text
Cannot load plugin opencv-bridge/0.1.3 (linux-x86_64): libopencv_core.so.406: cannot open shared object file
```

Build the plugin against your installed OpenCV instead. It needs a Rust toolchain, OpenCV development headers, Clang and pkg-config:

```sh
# Fedora:          sudo dnf install opencv-devel clang-devel llvm-devel pkgconf-pkg-config
# Debian / Ubuntu: sudo apt install libopencv-dev clang libclang-dev llvm pkg-config
git clone https://github.com/JiraPit/Morflow.git && cd Morflow
bash scripts/build-plugins.sh opencv-bridge
export MORFLOW_PLUGINS_PATH="$PWD/target/release/plugins"
```

The build is packaged as the version in `plugins/opencv-bridge/Cargo.toml`, so check out the release your pipeline declares. Set `MORFLOW_PLUGINS_PATH` in every process that runs the pipeline; it replaces the default plugin cache.

### Conda and Anaconda Python

Conda Python environments ship their own, older C++ runtime and load it ahead of the system one. If loading fails with an error mentioning `CXXABI_` or `GLIBCXX_`, that runtime is older than the one your OpenCV needs. Use the system Python in a virtual environment, or install OpenCV from conda-forge into the same environment so both use Conda's runtime.

## Development builds

For an installation without pkg-config, configure `OPENCV_INCLUDE_PATHS`, `OPENCV_LINK_PATHS`, and `MORFLOW_OPENCV_LINK_LIBS` (comma-separated `dylib=<library>` names). These are build settings. Runtime selection is controlled by the pipeline declaration and verified plugin cache, rather than a bridge-path environment variable.

`morflow_opencv_process` accepts contiguous image buffers and an operation descriptor defined in the plugin’s `src/lib.rs` and mirrored by each action’s private `src/opencv.rs` module. Inputs are borrowed and outputs use action-owned storage. OpenCV exceptions and Rust panics become errors at this boundary. See [the action guide](../../actions/image_opencv/README.md) for layouts, numerical behavior and supported operations.

## Interface maintenance

`src/interface.rs` defines this plugin's native interface. The plugin and its actions
compile that same source file; it contains declarations rather than processing code.
Changes to the interface require rebuilding its consuming actions. If their current
versions are already published, bump those action versions before releasing the change.
