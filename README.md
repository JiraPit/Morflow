# Morflow

**Morflow** is a modular, high-performance dataflow pipeline engine in Rust designed for high-throughput stream processing, digital signal processing (DSP), tensor computing, and media pipelines.

Morflow compiles and executes dataflow pipelines declared in human-readable `.morf` files. Processing steps (**Actions**) are dynamically loaded from precompiled shared libraries (`.so` / `.dll`) across a zero-copy, ABI-stable FFI boundary without recompiling the host application.

---

## Key Features

- **Dynamic Shared Library Plugins (`cdylib`):** Load, execute, and swap compiled actions at runtime with zero host recompilation and zero `dlopen`/`dlsym` overhead during execution.
- **Custom Dataflow DSL (`.morf`):** Expressive, stream-oriented syntax parsed with Chumsky parser combinators.
- **Rayon Multi-Core Auto-Parallelism:** Automatic DAG flow dependency discovery and work-stealing parallel dispatch across CPU cores.
- **Zero-Copy Multi-Dimensional Tensor Views:** $O(1)$ offset/stride slicing along arbitrary dimensions (`$tensor[0:240000]`, `$tensor[axis=1]`, `[1:3, :]`) without memory reallocations.
- **Mandatory Named Slice Loops (`each ($var)`):** Auto-parallelized dimension unrolling across worker threads with zero-copy layered environments and parallel tensor restacking.
- **Strict Static Validation:** Compile-time Single-Assignment (SSA) validation rejecting duplicate writes across flows (`$var is written to by multiple flows: <flow1>, <flow2>`).
- **Explicit Host Return (`emit`):** Emit single or multiple named payloads to the host application using `>> emit` or `>> emit("name")`.

---

## Core Terminology

| Term | Definition |
| :--- | :--- |
| **Pipeline** | A complete `.morf` document/file defining input parameter declarations (`accept $param`) and statements. Exactly **1 pipeline** is defined per `.morf` file. |
| **Flow** | An individual transformation chain connected with `>>` (e.g. `$source >> action_a >> $target_var`). |
| **Sub-Flow** | An inner flow chain enclosed within an `each` loop, `if/else` branch, or `route` block. |
| **Action** | A compiled dynamic library plugin (`cdylib`) implementing `extern "C" fn process(Payload) -> Payload`. |
| **Emit** | The terminal flow action that emits processed payloads from the pipeline back to the host application. |

---

## Architecture & Workspace Crates

```text
morflow/
├── core_types/       # ABI-stable types (Tensor, Payload, ActionArgs) & parallel tensor math
├── parser/           # Chumsky parser combinator & AST for .morf pipeline documents
├── pipeline/         # Host engine, DAG validator, plugin registry, & Rayon scheduler
├── actions/          # High-performance in-memory DSP & tensor dynamic plugins (.so / .dll)
│   ├── gain/         # Linear & dB amplitude scaling with SIMD/Rayon parallelism
│   ├── normalize/    # Peak & RMS loudness normalization
│   ├── biquad_filter/# Direct Form II Transposed IIR EQ filters (lowpass, highpass, bandpass, notch, peaking)
│   ├── compressor/   # Feedforward dynamic range compressor with soft knee & ballistics
│   ├── limiter/      # Fast brickwall peak limiter & tanh saturation soft clip
│   ├── noise_gate/   # Downward expansion noise gate with attack/hold/release state tracking
│   ├── stereo_widen/ # Mid/Side stereo field widen & center gain adjustment
│   ├── resample/     # Bandlimited windowed sinc / polyphase sample rate converter
│   ├── stft/         # Radix-2 Cooley-Tukey FFT short-time Fourier transform spectrograms
│   ├── delay/        # Circular ring buffer fractional delay with feedback & wet/dry mix
│   └── identity/     # Zero-overhead FFI pass-through action
└── examples/         # Sample .morf pipeline definitions
```

- **`core_types` ([`core_types/src/lib.rs`](file:///home/jira_pit/Documents/morflow/core_types/src/lib.rs)):** Shared vocabulary across the FFI boundary using `abi_stable`. Includes `Tensor` with zero-allocation parallel reductions (`peak_abs`, `rms`, `mean`), `Audio` with speaker layouts (`Mono`, `Stereo`, `Surround5_1`, etc.) and sample rate metadata, `Image` supporting color spaces (`RGB`, `RGBA`, `BGR`, `Grayscale`, etc.) and memory layouts (`HWC`, `CHW`), and `Payload` enums.
- **`parser` ([`parser/src/lib.rs`](file:///home/jira_pit/Documents/morflow/parser/src/lib.rs)):** Robust AST generator for parameter declarations, flow chains, variable taps, tensor slices, and loops.
- **`pipeline` ([`pipeline/src/lib.rs`](file:///home/jira_pit/Documents/morflow/pipeline/src/lib.rs)):** High-performance host runtime that executes `.morf` pipelines, caches loaded plugin symbols, and returns structured `PipelineOutputs`.
- **`actions` ([`actions/`](file:///home/jira_pit/Documents/morflow/actions)):** Standard library of 20+ in-memory DSP and Image Processing actions compiled as high-performance dynamic shared libraries (`cdylib`).

---

## Standard In-Memory Actions

All standard actions operate on in-memory `Payload::Image`, `Payload::Audio`, `Payload::Tensor`, or `Payload::Data` buffers without performing file I/O (which is handled by the host application).

### 1. Image Processing Actions

| Action | Description | Key Parameters |
| :--- | :--- | :--- |
| **`to_tensor`** | Standardizes any `Image` (or `Audio`/`Tensor`) to a canonical working `Tensor` | `color="rgb" \| "rgba" \| "bgr" \| "grayscale"`, `dtype="f32" \| "u8"`, `layout="hwc" \| "chw"`, `normalize=true` |
| **`resize`** | High-performance 2D spatial scaling with Rayon parallelism | `width=512`, `height=512`, `scale=0.5`, `filter="bilinear" \| "nearest" \| "bicubic" \| "area"`, `keep_aspect_ratio=false` |
| **`crop`** | Zero-copy $O(1)$ Region of Interest (ROI) bounding box cropping | `x=0`, `y=0`, `width=256`, `height=256` |
| **`pad`** | Canvas border extension with multiple boundary modes | `top=10`, `bottom=10`, `left=10`, `right=10`, `pad=16`, `mode="constant" \| "edge" \| "reflect"`, `fill=0.0` |
| **`color_adjust`** | Fused vectorized single-pass color grading | `brightness=0.1`, `contrast=1.2`, `gamma=1.0`, `saturation=1.1`, `exposure=0.0` |
| **`gaussian_blur`** | Pure separable $O(2K)$ 2D spatial Gaussian and Box smoothing | `sigma=1.5`, `radius=3`, `mode="gaussian" \| "box"` |
| **`edge_detect`** | Spatial derivative edge detection filters | `mode="sobel" \| "sobel_x" \| "sobel_y" \| "laplacian" \| "prewitt"`, `strength=1.0` |
| **`sharpen`** | Unsharp masking detail enhancement | `strength=1.5`, `sigma=1.0`, `radius=1` |
| **`threshold`** | Pixel intensity segmentation and automatic Otsu thresholding | `threshold=0.5`, `max_val=1.0`, `mode="binary" \| "binary_inv" \| "otsu" \| "truncate" \| "to_zero"` |
| **`rotate`** | Fast 90°/180°/270° CW and arbitrary 2D affine rotation | `angle=90.0`, `expand=true`, `fill=0.0` |
| **`flip`** | Coordinate axis reflection / mirroring | `axis="horizontal" \| "vertical" \| "both"` (or positional `"horizontal"`, `"vertical"`) |
| **`blend`** | Multi-layer compositing & blending modes | `mode="alpha" \| "multiply" \| "screen" \| "overlay" \| "add" \| "difference"`, `opacity=0.8`, `color="1.0,0.5,0.0"` |
| **`morphology`** | Separable morphological operations (Dilation, Erosion, Open, Close, Gradient) | `op="dilate" \| "erode" \| "open" \| "close" \| "gradient"`, `kernel_size=3`, `shape="rect" \| "cross" \| "ellipse"` |
| **`to_image`** | Clamps, quantizes, and formats working `Tensor` back to typed `Image` | `color="rgba" \| "rgb" \| "grayscale"`, `dtype="u8" \| "f32"`, `layout="hwc" \| "chw"`, `denormalize=true` |

### 2. Audio & DSP Actions

| Action | Description | Key Parameters |
| :--- | :--- | :--- |
| **`gain`** | Linear or decibel amplitude scaling | `db=+3.5`, `linear=1.5` |
| **`normalize`** | Peak or RMS loudness normalization | `target_peak=1.0`, `target_peak_db=-0.1`, `mode="peak" \| "rms"` |
| **`biquad_filter`** | RBJ Direct Form II Transposed IIR EQ filters | `type="lowpass" \| "highpass" \| "bandpass" \| "notch" \| "peaking"`, `freq=1000.0`, `q=0.707`, `gain_db=0.0`, `sample_rate=44100` |
| **`compressor`** | Feedforward dynamic range compressor | `threshold_db=-12.0`, `ratio=4.0`, `attack_ms=10.0`, `release_ms=100.0`, `knee_db=2.0`, `makeup_db=0.0` |
| **`limiter`** | Fast brickwall peak limiter & soft clipper | `ceiling_db=-0.1`, `release_ms=50.0`, `mode="brickwall" \| "soft_clip"`, `drive=1.0` |
| **`noise_gate`** | Downward expansion noise gate | `threshold_db=-45.0`, `ratio=10.0`, `attack_ms=2.0`, `hold_ms=10.0`, `release_ms=50.0` |
| **`stereo_widen`** | Mid/Side stereo field widening | `width=1.2` (0 = mono, 1 = unchanged, >1 = wider), `center_gain_db=0.0` |
| **`resample`** | Bandlimited windowed sinc resampling | `from_rate=48000.0`, `to_rate=44100.0` |
| **`stft`** | Radix-2 FFT Short-Time Fourier Transform | `n_fft=1024`, `hop_size=256` (outputs 3D Spectrogram Tensor `[channels, n_fft/2+1, frames]`) |
| **`delay`** | Circular ring buffer echo delay | `time_ms=120.0`, `feedback=0.35`, `mix=0.3` |

---

## The `.morf` Pipeline Language

### 1. Parameter Declarations (`accept`)
Pipelines begin by accepting input arguments with optional default values:
```morf
accept $input_audio
accept $sample_rate = 44100
accept $target_lufs = -14.0
```

### 2. Flow Chaining & Variable Tapping
Data moves left-to-right through actions using the `>>` stream operator. Intermediate payloads can be tapped into named variables:
```morf
load_audio("input.flac") >> $raw_audio

# Tapped variable used in downstream flow
$raw_audio >> denoise >> $clean_audio
```

### 3. Zero-Copy Tensor Slicing
Extract slices across ranges or specific dimensions without copying memory buffers:
```morf
# Slice first 240,000 samples (5 seconds at 48kHz)
$raw_audio[0:240000] >> compute_noise_profile >> $noise_profile

# Slice along specific dimension index
$multichannel_audio[axis=1] >> normalize >> $mono_preview
```

### 4. Auto-Parallel Slice Loops (`each`)
Iterate over tensor slices in parallel with mandatory named iteration variables:
```morf
$mastered[axis=1] >> each ($channel) {
    $channel
        >> highpass(freq=30)
        >> stereo_widen(amount=1.2)
} >> $final_mix
```
All slices execute concurrently across Rayon worker threads and automatically restack into a contiguous tensor along the sliced axis.

### 5. Conditional Branching & Metrics
Evaluate tensor metrics (`peak`, `rms`, `mean`, `len`) inline:
```morf
$raw_audio
    >> denoise(profile=$noise_profile)
    >> if ($raw_audio.peak > 0.0) {
        compressor(ratio=4.0, attack_ms=15) >> soft_clip
    } else {
        normalize(target_lufs=-14.0)
    }
    >> $mastered
```

### 6. Returning to Host (`emit`)
Flows return outputs to the host application using `emit`. `emit` functions both as a **terminal return** and as a **mid-stream zero-copy pass-through** (stream tap):

- **Terminal single output:**
  ```morf
  $final_mix >> export("master_out.wav") >> emit
  ```
- **Multiple named outputs across flows:**
  ```morf
  $audio[0:4] >> filter_low  >> emit("low_band")
  $audio[4:8] >> filter_high >> emit("high_band")
  ```
- **Mid-stream pass-through tap (intermediate previews/checkpoints):**
  ```morf
  $img
      >> resize(width=512, height=512)
      >> emit("thumbnail")              # Emitted to host as outputs["thumbnail"]
      >> color_adjust(contrast=1.2)
      >> gaussian_blur(sigma=2.0)
      >> emit("final_blur")             # Emitted to host as outputs["final_blur"]
  ```

---

## Examples & Rust Host Crates

Morflow includes end-to-end Rust host examples under [`examples/rust/`](file:///home/jira_pit/Documents/morflow/examples/rust):

### 1. Image Processing Host Crate
[`examples/rust/image_processing`](file:///home/jira_pit/Documents/morflow/examples/rust/image_processing):
- Host loads and decodes `input.png` with the `image` crate.
- Compiles neighboring [`image_pipeline.morf`](file:///home/jira_pit/Documents/morflow/examples/rust/image_processing/image_pipeline.morf) using `Morflow::load`.
- Executes zero-copy tensor transformations, resizing, color grading, and gaussian blur.
- Encodes output payload to `output.png`.

```bash
cargo run --package image_processing
```

[`image_pipeline.morf`](file:///home/jira_pit/Documents/morflow/examples/rust/image_processing/image_pipeline.morf):
```morf
accept $img_in
accept $target_width = 512
accept $target_height = 512

# Direct chained dataflow without temporary variables
$img_in
    >> to_tensor(color="rgb", dtype="f32", layout="hwc", normalize=true)
    >> resize(width=$target_width, height=$target_height, filter="bilinear")
    >> color_adjust(contrast=1.15, saturation=1.05, brightness=0.02)
    >> gaussian_blur(sigma=1.2)
    >> to_image(color="rgba", dtype="u8")
    >> emit
```

### 2. Audio DSP Processing Host Crate
[`examples/rust/audio_processing`](file:///home/jira_pit/Documents/morflow/examples/rust/audio_processing):
- Host decodes `input.wav` using `hound`.
- Compiles neighboring [`audio_pipeline.morf`](file:///home/jira_pit/Documents/morflow/examples/rust/audio_processing/audio_pipeline.morf).
- Executes DSP pipeline with multi-channel Rayon parallelization (`resample`, `normalize`, `each ($channel) { biquad_filter >> compressor }`, `stereo_widen`, `limiter`).
- Encodes output audio stream to `output.wav`.

```bash
cargo run --package audio_processing
```

### 3. Multi-Channel Audio Split Host Crate
[`examples/rust/audio_split`](file:///home/jira_pit/Documents/morflow/examples/rust/audio_split):
- Host loads multi-channel WAV audio.
- Compiles neighboring [`audio_split_pipeline.morf`](file:///home/jira_pit/Documents/morflow/examples/rust/audio_split/audio_split_pipeline.morf).
- Extracts and processes individual channels in parallel and emits multiple named streams (`emit("left_filtered")`, `emit("right_filtered")`).
- Host saves each channel stream to separate output WAV files.

```bash
cargo run --package audio_split
```

---

## Python Interface (`morflow`)

Morflow provides high-performance Python bindings with zero-copy NumPy integration built with PyO3 and Maturin:

### Install Python Extension
```bash
cd bindings/python
maturin develop
```

### Python Host API
```python
import morflow
import numpy as np

# 1. Load pipeline
pipeline = morflow.load("image_pipeline.morf")

# 2. Execute with NumPy array input (zero-copy)
input_array = np.zeros((600, 800, 3), dtype=np.uint8)
output_array = pipeline.run(input_array)

print("Output shape:", output_array.shape)
```

### Python Examples
Run end-to-end Python host examples under [`examples/python/`](file:///home/jira_pit/Documents/morflow/examples/python):
```bash
python examples/python/image_processing/main.py
python examples/python/audio_processing/main.py
python examples/python/audio_split/main.py
```

---

## Rust Host API

Add Morflow crates to your `Cargo.toml` and execute pipelines directly:

```rust
use core_types::{ColorSpace, Image, Payload};
use pipeline::{Morflow, PipelineOutputs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load and compile pipeline from neighboring file or string
    let mut pipeline = Morflow::load("image_pipeline.morf")?;

    // 2. Prepare payload from decoded media buffer
    let raw_bytes = vec![255u8; 800 * 600 * 3];
    let img = Image::from_u8_hwc(&raw_bytes, 800, 600, ColorSpace::Rgb)?;

    // 3. Execute with auto-parallel Rayon scheduling
    let outputs: PipelineOutputs = pipeline.run(Payload::Image(img))?;

    // 4. Access emitted results
    let single_payload: Payload = outputs.into_single()?;

    Ok(())
}
```

---

## Creating Action Plugins (`cdylib`)

Actions are lightweight Rust libraries that compile to standard shared objects (`.so` on Linux, `.dll` on Windows, `.dylib` on macOS):

`actions/my_custom_action/Cargo.toml`:
```toml
[package]
name = "my_custom_action"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
core_types = { path = "../../core_types" }
```

`actions/my_custom_action/src/lib.rs`:
```rust
use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    match payload {
        Payload::Tensor(mut tensor) => {
            // High-performance, in-place or zero-copy transformation
            Payload::Tensor(tensor)
        }
        other => other,
    }
}
```

---

## Building & Testing

### Build Everything (Engine & Action Plugins)
Run the automated release build script:
```bash
./build-release.sh
```
This compiles all workspace crates and copies compiled `.so` / `.dll` plugins to `target/release/actions/`.

### Run Workspace Tests
```bash
cargo test --workspace
```

### Run Example CLI
```bash
./target/release/pipeline
```

### Plugin Discovery Path
By default, the engine searches for actions in:
1. Directory path configured in `MORFLOW_ACTIONS_PATH` environment variable.
2. `<executable_dir>/actions/` and `<executable_dir>/`.
3. `target/release/actions/` and `target/debug/actions/`.

To specify a custom action plugin folder:
```bash
export MORFLOW_ACTIONS_PATH="/path/to/custom/actions"
```
