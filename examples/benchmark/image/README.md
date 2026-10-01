# Image transformation benchmark

Compare an image-to-tensor pipeline implemented with **Pillow (PIL) + NumPy** and **Morflow's Python binding**. Each call starts with the same RGB image and performs:

1. Center crop to 75% of the input width and height (rounded down, minimum one pixel).
2. Horizontal flip.
3. Nearest-neighbor resize to the requested output dimensions.
4. Convert to a float32 HWC tensor with RGB values normalized to `[0, 1]`.

The benchmark checks output shape, dtype, and pixel values against the PIL/NumPy reference before measuring performance. Differences beyond an absolute tolerance of `1e-6` fail the run; a failed run does not write a results file.

## Setup

Use Python 3.9 or newer. From the repository root:

```sh
python -m pip install -r examples/benchmark/image/requirements.txt
```

The PIL-only benchmark does not need Morflow. For Morflow, install the current Python binding and prepare the exact action versions imported by `pipeline.morf`:

```sh
python -m pip install ./bindings/python
morflow prep examples/benchmark/image/pipeline.morf
```

For a local development checkout whose versions have not been released, build and package the release actions instead:

```sh
bash scripts/build-actions.sh
python scripts/prepare_test_actions.py --profile release
```

If your Python binding is installed in editable mode, refresh its native extension too:

```sh
maturin develop --release --manifest-path bindings/python/Cargo.toml
```

Building with `scripts/build-engine.sh` produces libraries in `target/release/`; it does not replace an older extension already installed in the Python package. Reinstalling the binding is required after engine/interface changes.

Then pass `--actions-path target/release/actions` when benchmarking. Use release builds when comparing speed. The pipeline imports `base/0.3.1` and `image_basics/0.3.1`; the binding and actions must come from a compatible checkout.

## Run

Commands work from any directory when you provide the script's path. These examples run from the repository root:

```sh
# Both backends (default)
python examples/benchmark/image/benchmark.py

# Choose one backend
python examples/benchmark/image/benchmark.py --pil
python examples/benchmark/image/benchmark.py --morflow --actions-path target/release/actions

# Explicitly choose both, supply a photo, and set the result path
python examples/benchmark/image/benchmark.py --pil --morflow \
  --image photo.png --iterations 100 --samples 7 --warmup 10 \
  --output results/image-benchmark.json

# Control generated input dimensions, output dimensions, and Morflow workers
python examples/benchmark/image/benchmark.py --morflow \
  --width 1920 --height 1080 --output-width 512 --output-height 512 \
  --threads 4 --actions-path target/release/actions
```

Without `--image`, the script generates deterministic RGB pixels using `--seed` (default `0`). Input defaults to 1920×1080, output to 512×512. With `--image`, the file's dimensions take precedence over `--width` and `--height`.

Defaults are five warmup calls per backend, five timing samples, and 50 transformations per sample. Samples alternate backend order. `--threads` sets `RAYON_NUM_THREADS` before loading Morflow; without it, the existing environment or Rayon defaults apply. Pillow and NumPy use their own implementations; this is not a comparison with identical threading.

## What is timed

Timing uses `perf_counter_ns`. It measures complete transformation calls, including Python/native call overhead, Morflow's input conversion, output allocation/materialization, and output release. The pipeline and reusable source objects are prepared once. Image decoding, image generation, dependency loading, correctness checks, and warmup calls are excluded.

The result is a practical comparison of these two Python-callable implementations. It is not a measurement of isolated native action execution. Morflow runs its normal runtime shape checks and output verification.

The nearest resize uses `floor(output_index × input_size / output_size)` with float32 coordinates. The PIL backend uses Pillow for crop/flip and NumPy for this resize and normalization. Pillow's built-in nearest resize uses a different pixel-center convention, so substituting it would change the workload's output. This benchmark does not cover bilinear resize, color adjustment, or blur.

## JSON results

By default, each run writes a timestamped file under `examples/benchmark/image/results/` (ignored by Git). `--output` selects another location and replaces that file after a successful run. Files are written atomically.

The report contains:

- `schema_version`, UTC start time, input source/seed, dimensions, crop, transformation steps, and pipeline source.
- Measurement settings and timing scope.
- Python, package, OS, CPU, action-cache, and thread settings.
- Per-backend output validation and maximum absolute error.
- Raw sample durations in nanoseconds and sample mean milliseconds per image.
- Mean, median, minimum, maximum, and standard deviation **of sample means**, plus overall images per second.
- For both backends, PIL time divided by Morflow time; a ratio above 1 means Morflow was faster for this run.

These are sample averages, not individual-call latency percentiles. Compare runs on the same machine, with the same input and settings, and keep other system activity low. The benchmark makes no assumption that either implementation will be faster.

## Verify the suite

```sh
python -m unittest discover -s examples/benchmark -t examples/benchmark -p 'test_*.py'

# Also exercise Morflow across small, uneven, and upsampled inputs
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  python -m unittest discover -s examples/benchmark -t examples/benchmark -p 'test_*.py'
```

The native integration test runs only when `MORFLOW_ACTIONS_PATH` is set.


## Load once, run repeatedly

Both benchmarks separate setup from execution:

```text
Load/decode or generate the input once
Load each Morflow pipeline once; prepare the PIL/NumPy source once
Run validation → run warmup → run → run → run → …
```

For `--image`, the image is opened and decoded exactly once into an owned RGB NumPy array. The file is closed before either backend runs. PIL reuses an in-memory image; Morflow reuses a typed wrapper over the same decoded pixels. Iterations never reopen or decode the file. The tensor benchmark similarly generates its input once and loads its checked-in pipeline.morf file once.

Every iteration transforms the original input, rather than feeding the previous result back in. Input and output conversions needed by the Python binding still occur within `pipeline.run`; pipeline loading and disk I/O do not.
