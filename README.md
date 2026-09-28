# Morflow

> **Modular, High-Performance Media & Tensor Dataflow Engine**

Morflow combines the speed of **Rust** and the flexibility of **Python** with an expressive stream DSL (`.morf`). Write concise media and tensor pipelines, swap dynamically compiled action plugins at runtime, and execute across CPU cores with auto-parallelism and zero-copy tensor slicing.

---

## ⚡ Quickstart

### 1. Define your pipeline (`pipeline.morf`)
```morf
accept $img
$img
    >> to_tensor(color="rgb", normalize=true)
    >> resize(width=512, height=512, filter="bilinear")
    >> color_adjust(contrast=1.2, saturation=1.1)
    >> gaussian_blur(sigma=1.0)
    >> to_image
    >> emit
```

### 2. Run in Python (Zero-Copy NumPy)
```python
import morflow
import numpy as np

# Load and run pipeline with zero-copy NumPy buffers
pipeline = morflow.load("pipeline.morf")
output_img = pipeline.run(input_np_array)
```

### 3. Or run in Rust
```rust
use pipeline::{Morflow, Payload};

let mut pipeline = Morflow::load("pipeline.morf")?;
let output = pipeline.run(Payload::Image(image))?.into_single()?;
```

---

## ✨ Key Features

- **🚀 Zero-Copy Tensor Engine**: Multi-dimensional tensor slicing ($O(1)$ views, strided slices, planar audio & image channels) without memory reallocations.
- **🔌 Dynamic Shared Library Plugins (`cdylib`)**: Actions are loaded from `.so`/`.dll`/`.dylib` binaries at runtime. Add or swap algorithms with **zero host recompilation**.
- **⚡ Rayon Multi-Core Auto-Parallelism**: Automatic dependency tracking executes independent pipeline flows and tensor loops (`each ($channel)`) concurrently across worker threads.
- **🐍 First-Class Python Bindings**: Seamless PyO3 + NumPy integration. Zero memory copies between Python and Rust.
- **📋 25 Standard Actions Included**: Out-of-the-box support for image processing, audio DSP, equalizers, dynamics, spectral STFT, and format conversions.

---

## 🛠️ Installation & Building

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (edition 2021+)
- Python 3.8+ with `numpy` (optional, for Python bindings)

### Build Everything (Engine + 25 Actions)
```bash
./build-release.sh
```
This builds the core engine, CLI, and compiles all action plugins into `target/release/actions/`.

### Install Python Bindings
```bash
cd bindings/python
maturin develop --release
```

---

## 📦 Standard Action Library (25 Actions)

Each action is a standalone dynamic plugin accompanied by a strict technical specification in its directory:

### 🖼️ Image Processing (12)
| Action | Spec | Description |
| :--- | :--- | :--- |
| **`resize`** | [`SPEC.md`](actions/resize/SPEC.md) | Spatial resampling (Nearest, Bilinear, Bicubic, Area Box). |
| **`crop`** | [`SPEC.md`](actions/crop/SPEC.md) | Bounding box spatial cropping & sub-tensor slicing. |
| **`pad`** | [`SPEC.md`](actions/pad/SPEC.md) | Border padding (constant, edge clamp, reflect). |
| **`color_adjust`** | [`SPEC.md`](actions/color_adjust/SPEC.md) | Brightness, contrast, gamma, saturation, exposure tone mapping. |
| **`gaussian_blur`** | [`SPEC.md`](actions/gaussian_blur/SPEC.md) | Separable 1D Gaussian & box blur smoothing. |
| **`edge_detect`** | [`SPEC.md`](actions/edge_detect/SPEC.md) | Sobel, Prewitt, and Laplacian spatial gradient filters. |
| **`sharpen`** | [`SPEC.md`](actions/sharpen/SPEC.md) | High-frequency detail unsharp masking. |
| **`threshold`** | [`SPEC.md`](actions/threshold/SPEC.md) | Binary, inverted, truncate, and Otsu binarization. |
| **`rotate`** | [`SPEC.md`](actions/rotate/SPEC.md) | Affine 2D rotation with canvas expansion. |
| **`flip`** | [`SPEC.md`](actions/flip/SPEC.md) | Horizontal and vertical spatial tensor mirroring. |
| **`blend`** | [`SPEC.md`](actions/blend/SPEC.md) | Alpha, multiply, screen, and overlay compositing. |
| **`morphology`** | [`SPEC.md`](actions/morphology/SPEC.md) | Dilation, erosion, opening, closing, and gradient operators. |

### 🔊 Audio & DSP (10)
| Action | Spec | Description |
| :--- | :--- | :--- |
| **`gain`** | [`SPEC.md`](actions/gain/SPEC.md) | Linear amplitude and decibel gain scaling. |
| **`normalize`** | [`SPEC.md`](actions/normalize/SPEC.md) | Multi-channel peak & RMS level normalization. |
| **`biquad_filter`** | [`SPEC.md`](actions/biquad_filter/SPEC.md) | RBJ Direct Form II Transposed IIR EQ filters. |
| **`compressor`** | [`SPEC.md`](actions/compressor/SPEC.md) | Soft-knee dynamic compressor with makeup gain. |
| **`limiter`** | [`SPEC.md`](actions/limiter/SPEC.md) | Brickwall peak limiter & soft-clipping saturation. |
| **`noise_gate`** | [`SPEC.md`](actions/noise_gate/SPEC.md) | Downward expander with hold and release envelopes. |
| **`stereo_widen`** | [`SPEC.md`](actions/stereo_widen/SPEC.md) | Mid/Side stereo field widening & center channel balance. |
| **`resample`** | [`SPEC.md`](actions/resample/SPEC.md) | Windowed sinc band-limited sample rate conversion. |
| **`stft`** | [`SPEC.md`](actions/stft/SPEC.md) | Radix-2 Cooley-Tukey FFT magnitude spectrograms. |
| **`delay`** | [`SPEC.md`](actions/delay/SPEC.md) | Ring buffer feedback delay line with wet/dry mix. |

### ⚙️ Core & Utility (3)
| Action | Spec | Description |
| :--- | :--- | :--- |
| **`identity`** | [`SPEC.md`](actions/identity/SPEC.md) | Zero-overhead pass-through action for routing/debugging. |
| **`to_tensor`** | [`SPEC.md`](actions/to_tensor/SPEC.md) | Standardizes images/audio to unified multidimensional tensors. |
| **`to_image`** | [`SPEC.md`](actions/to_image/SPEC.md) | Converts tensors into strongly-typed `Image` payloads. |

---

## 📖 The `.morf` Pipeline Language

Pipelines in Morflow are declared using concise stream syntax:

### Flow Chaining & Taps
```morf
accept $input_audio

# Stream left-to-right with >> and tap into variables
$input_audio >> normalize >> $clean_audio
$clean_audio >> compressor(ratio=4.0) >> emit
```

### Auto-Parallel Tensor Slicing (`each`)
```morf
# Processes each channel concurrently on Rayon worker threads
$audio[axis=1] >> each ($channel) {
    $channel >> biquad_filter(type="highpass", freq=80)
} >> $filtered_audio
```

### Multiple & Mid-Stream Emits
```morf
$img
    >> resize(width=256, height=256)
    >> emit("thumbnail")  # Intermediate tap emitted to host
    >> color_adjust(contrast=1.2)
    >> emit("processed")  # Final output emitted to host
```

---

## 🧩 Writing a Custom Action Plugin

Creating new actions takes just a few lines of Rust:

```rust
// actions/custom_invert/src/lib.rs
use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType { DataType::Tensor }

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType { DataType::Tensor }

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    match payload.unwrap_payload() {
        Payload::Tensor(mut tensor) => {
            // Apply custom transformation in-place
            Payload::Tensor(tensor)
        }
        other => other.clone(),
    }
}
```
Compile it to `.so`/`.dll` with `cargo build --release` and drop it into your actions directory—Morflow discovers and executes it immediately.

---

## 📂 Ready-to-Run Examples

Explore complete host implementations in both Rust and Python:

- [`examples/rust/image_processing/`](examples/rust/image_processing): Image resizing, color grading, and blur.
- [`examples/rust/audio_processing/`](examples/rust/audio_processing): Audio mastering with multi-channel DSP.
- [`examples/rust/audio_split/`](examples/rust/audio_split): Parallel stereo splitting and multiple named emits.
- [`examples/python/`](examples/python): Matching Python examples with NumPy integration.

---

## 🧪 Testing

```bash
# Run all workspace unit & pipeline integration tests
cargo test --workspace

# Run Python bindings tests
pytest bindings/python/tests
```

---

## 📄 License

MIT License. See [`LICENSE`](LICENSE) for details.
