# Shared OpenCV image actions

`image_opencv/0.1.0` accelerates six spatial operations using the execution machine's shared OpenCV `core` and `imgproc` libraries:

| Action | OpenCV implementation |
| --- | --- |
| `rotate` | Quarter turns and affine resampling |
| `resize` | Nearest, linear, cubic or area resampling |
| `gaussian_blur` | Gaussian or box filtering |
| `morphology` | Dilation, erosion, opening, closing or gradient |
| `edge_detect` | Sobel, Prewitt or Laplacian filtering |
| `sharpen` | Gaussian unsharp masking |

Keep `crop`, `pad`, `flip`, `to_image`, `blend`, `color_adjust` and `threshold` in `image_basics`. Views and simple pixel arithmetic do not justify a second implementation in this pack.

## Install and use

Declare the plugin alongside the action imports:

```perl
plugin opencv-bridge/0.1.0
from image_opencv/0.1.0 import resize
```

```sh
morflow prep examples/opencv/pipeline.morf
morflow plugins list
morflow check examples/opencv/pipeline.morf
```

Preparation installs a prebuilt plugin. Install its compatible shared OpenCV libraries through the execution machine's package manager. Users do not compile the bridge. OpenCV and its C++ runtime must be discoverable by the system loader.

For development builds, install shared OpenCV development libraries, Clang, a C++17 compiler and pkg-config:

```sh
bash scripts/build-plugins.sh opencv-bridge
bash scripts/build-actions.sh image_opencv
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" morflow check examples/opencv/pipeline.morf
```

The plugin uses the Rust `opencv` crate to generate bindings during its build. Only the adapter is packaged; OpenCV remains dynamically linked. Building actions requires neither OpenCV headers nor libraries. See [the plugin guide](../../plugins/README.md) for versions, cache configuration and custom plugin development, and [the OpenCV plugin guide](../../plugins/opencv-bridge/README.md) for its system dependencies.

## Shape checking and execution

Static checks and pipeline loading use pure shape and argument analysis, without opening the bridge. Missing OpenCV is reported by `process` when the pipeline runs. There is no fallback to basics. Failed library loads can be retried; each loaded pipeline retains its selected plugin and successful library handle until it is dropped.

The engine calls `shapecheck` with actual dimensions before every execution. Shape checking parses processing options into the execution plan, so `process` reuses them. Known dimensions outside OpenCV's signed 32-bit range are rejected; rotation dimensions must also be below 32767; dynamic dimensions remain available for runtime checking.

These actions accept U8 or F32 tensors with `[height,width]`, HWC, or CHW shapes. Resize and rotation support up to four channels; the other operations support up to 512. Raw tensors follow Morflow's HWC-first convention: a final dimension of at most four is interpreted as channels; otherwise an initial dimension of at most four indicates CHW. CHW inputs are converted to interleaved storage for OpenCV and returned in CHW order. Non-contiguous views are materialized only when necessary. Contiguous HWC input is borrowed, and OpenCV writes into the caller-owned output allocation. Input buffers remain unchanged. Rank, channels and dtype are preserved, including a singleton channel dimension.

## Pixel behavior

Filters use replicated borders to match basics. Morphology preserves basics' square, cross and disk masks, including the effective odd mask size `2 * floor(kernel_size / 2) + 1`. Gradient uses one dilation and erosion, as basics does. A kernel size of zero or one is a no-op for every morphology mode, matching basics. Directional Sobel and Laplacian return absolute responses. F32 filters retain their numerical range; U8 results saturate to `[0,255]`. Floating-point accumulation and U8 rounding can differ slightly between implementations.

Resize uses OpenCV's interpolation definitions. Nearest selection, cubic coefficients and area integration can differ from basics. Shape prediction and argument names match basics, but resize output is not promised to be pixel-identical. Width/height, scales and aspect-ratio fitting use the engine's predicted output dimensions.

Rotation follows basics' clockwise angle convention and canvas dimensions, including swapping width and height for quarter turns even when `expand=false`. Quarter turns preserve pixels exactly; arbitrary angles use OpenCV bilinear affine sampling and constant borders, which can differ from basics at edges and in interpolation precision. CHW inputs keep CHW output.

## Performance and threading

OpenCV chooses optimizations according to its installed build and available CPU. Actual acceleration depends on image dimensions, operation, layout conversions and thread settings; benchmark your workload. This backend does not change OpenCV's global thread settings. Configure the installation's OpenCV parallel backend before starting the process, and account for Morflow scheduling independent flows concurrently.

## Validation

```sh
MORFLOW_PLUGINS_PATH="$PWD/target/release/plugins" \
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  cargo test -p pipeline --test opencv_actions
```

Official operation references: [filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) and [resampling](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
