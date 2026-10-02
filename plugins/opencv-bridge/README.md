# OpenCV bridge plugin 0.1.0

This prebuilt Rust plugin connects `image_opencv` actions to shared system OpenCV. It uses the Rust `opencv` crate with only the `core` and `imgproc` bindings. OpenCV is not bundled or statically linked.

Declare `plugin opencv-bridge/0.1.0` in the pipeline and run `morflow prep`. Users install the compatible OpenCV shared libraries through their system package manager; users do not compile the adapter when a prebuilt artifact is available.

Published Linux artifacts are built on Ubuntu 24.04 against its OpenCV development package. macOS artifacts use Homebrew OpenCV. Windows artifacts use shared OpenCV from vcpkg's `x64-windows` triplet. The execution environment must provide the corresponding shared-library ABI and a compatible C++ runtime.

For local development, install OpenCV development headers, shared libraries, Clang, a C++17 compiler and pkg-config, then run:

```sh
bash scripts/build-plugins.sh opencv-bridge
```

For an installation without pkg-config, configure `OPENCV_INCLUDE_PATHS`, `OPENCV_LINK_PATHS`, and `MORFLOW_OPENCV_LINK_LIBS` (comma-separated `dylib=<library>` names). These are build settings. Runtime selection is controlled by the pipeline declaration and verified plugin cache, rather than a bridge-path environment variable.

`morflow_opencv_process` accepts contiguous image buffers and an operation descriptor defined by `morflow-opencv`. Inputs are borrowed and outputs use action-owned storage. OpenCV exceptions and Rust panics become errors at this boundary. See [the action guide](../../backends/opencv/README.md) for layouts, numerical behavior and supported operations.
