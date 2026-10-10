# Image transformation benchmark

Compare **PIL/NumPy**, **morf-basic**, and **morf-opencv** using the same in-memory RGB image. Each execution performs:

1. Center crop to 75% of the input width and height.
2. Horizontal flip.
3. Nearest-neighbor resize to the requested output size.
4. Convert to a float32 HWC tensor normalized to `[0,1]`.

`morf-basic.morf` uses `image_basics` for crop, flip and resize. `morf-opencv.morf` uses the same crop and flip actions and replaces resize with `image_opencv/0.1.2`. Both use `base/0.3.2` for tensor conversion. This workload compares OpenCV resize within the existing pipeline; it does not measure the pack's blur, rotation, morphology, edge detection or sharpening actions.

## Setup

From the repository root:

```sh
python3 -m pip install -r examples/benchmark/image/requirements.txt
python3 -m pip install ./bindings/python
bash scripts/build-actions.sh
```

For the OpenCV backend, install OpenCV 4 development headers and shared core/imgproc libraries, a C++17 compiler and pkg-config, then build the plugin for local development:

```sh
bash scripts/build-plugins.sh opencv-bridge
export MORFLOW_PLUGINS_PATH="$PWD/target/release/plugins"
```

For published artifacts, `morflow prep examples/benchmark/image/morf-opencv.morf` installs the prebuilt plugin; only compatible runtime OpenCV libraries are needed. OpenCV remains a shared system dependency; it is not bundled into the actions. See [the action guide](../../../actions/image_opencv/README.md) for other platforms and library discovery. Python's C++ runtime must be compatible with the installed OpenCV, particularly when mixing Conda and system libraries. An error mentioning `GLIBCXX` indicates an incompatible runtime; use compatible installations rather than silently substituting a backend.

Use `--actions-path target/release/actions` for locally built, prepared actions. For published versions, `morflow prep` can prepare either pipeline file instead. The PIL-only run does not require Morflow or OpenCV.

If the installed binding predates your checkout's engine interface, rebuild it:

```sh
maturin develop --release --manifest-path bindings/python/Cargo.toml
```

## Run

```sh
# All three backends; no selection flags means all.
python3 examples/benchmark/image/benchmark.py --actions-path target/release/actions

# Select one or any combination.
python3 examples/benchmark/image/benchmark.py --pil
python3 examples/benchmark/image/benchmark.py --morf-basic --actions-path target/release/actions
python3 examples/benchmark/image/benchmark.py --morf-opencv --actions-path target/release/actions

# Compare both Morflow variants using a photo.
python3 examples/benchmark/image/benchmark.py --morf-basic --morf-opencv \
  --image photo.png --iterations 100 --samples 7 --warmup 10 \
  --actions-path target/release/actions --output results/image-benchmark.json

# Change input/output sizes and Morflow workers.
python3 examples/benchmark/image/benchmark.py --pil --morf-basic --morf-opencv \
  --width 1920 --height 1080 --output-width 512 --output-height 512 \
  --threads 4 --actions-path target/release/actions
```

Input defaults to seeded RGB pixels at 1920×1080; output defaults to 512×512. `--seed` controls generation. With `--image`, file dimensions take precedence over generated dimensions. Defaults are five warmup calls, five samples and fifty transformations per sample. Backend order alternates between samples.

`--threads` sets Morflow's Rayon worker count before loading it. Configure OpenCV threading before starting Python; the benchmark records `OPENCV_FOR_THREADS_NUM` and the plugin cache path. OpenCV's available optimizations and threading depend on its installed build. This is a comparison of complete implementations, not necessarily equal thread counts.

## Load once, run repeatedly

The input file is opened and decoded exactly once into an owned RGB NumPy array, then closed. Without a file, pixels are generated once. PIL prepares one in-memory image. Each selected Morflow variant calls `morflow.load` exactly once on its checked-in `.morf` file and creates one reusable input wrapper.

```text
Load the image once → load each selected pipeline once → run → run → run → …
```

Validation, warmup and timed runs reuse these instances. Every iteration starts from the original input. Iterations do not read files, reload pipelines or decode images.

## Validation and timing

Before timing, output dimensions, float32 dtype and pixels are checked against independently calculated PIL/NumPy references. Pillow handles crop/flip; NumPy calculates nearest sampling and normalization. The timed PIL backend follows basics' float32 scale calculations. OpenCV's reference follows its double-precision inverse-scale calculation. At some resize ratios, these conventions select different neighboring pixels at rounding boundaries.

Each backend must match its own reference within `1e-6`. The report also records each backend's maximum pixel difference and number of differing pixels compared with PIL. The console reports OpenCV rounding differences when present. This distinguishes valid library sampling behavior from an incorrect transformation. Pillow's built-in nearest resize has a different pixel-center convention and is not substituted here. Failed validation stops the run without writing a report.

Timing uses `perf_counter_ns` and includes transformation calls, Python/native overhead, output materialization and release. Input decoding/generation, source setup, pipeline loading, reference calculation, validation and warmup are excluded. Morflow uses its normal runtime shape checks and output verification.

## JSON reports

Reports use schema version 2 and are saved under `image/results/` unless `--output` specifies a path. The writer replaces the selected file atomically after success. Reports include:

- Input source, dimensions, crop, transformation steps and both pipeline sources.
- Timing settings, scope, package versions, platform and thread/library configuration.
- Per-backend reference conventions, validation errors, differences from PIL and raw sample durations.
- Mean, median, minimum, maximum and standard deviation of sample means, plus images per second.
- A timing ratio for every selected backend pair, including `morf_basic_time_divided_by_morf_opencv_time` when both are selected.

A ratio greater than one means the denominator backend was faster in that run. These are sample means, not individual-call latency percentiles. Generated reports and artifacts under the benchmark directory are ignored by Git.

## Tests

```sh
python3 -m unittest discover -s examples/benchmark -t examples/benchmark -p 'test_*.py'

# Native tests, with prepared actions and the OpenCV adapter:
MORFLOW_PLUGINS_PATH="$PWD/target/release/plugins" \
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  python3 -m unittest discover -s examples/benchmark/image -t examples/benchmark
```

Native basics tests require `MORFLOW_ACTIONS_PATH`; the OpenCV variant is also exercised when `MORFLOW_PLUGINS_PATH` is set. Mocked full-run tests independently verify one image decode and one pipeline load per selected Morflow backend across validation, warmup and repeated runs.
