# Morflow

> **The Unified, High-Performance Media & Data Processing Engine for AI and Production Systems.**

Morflow is a modular dataflow engine designed to bridge the gap between AI experimentation and production deployment. By defining data transformations in clean, human-readable `.morf` pipeline files, Morflow allows teams to author data and media preprocessing once and execute it anywhere with native Rust performance and zero train-serve skew.

---

## 🎯 The Problem Morflow Solves

In modern machine learning and data engineering workflows, preprocessing code written in Python training notebooks is frequently re-implemented or copy-pasted into production services. This leads to subtle numerical drift (**train-serve skew**), performance bottlenecks, complex dependency bloat, and coupled release cycles.

Morflow solves this by decoupling the data pipeline from the host application:

- **🔒 Zero Train-Serve Skew**: Guarantees identical, deterministic mathematical operations across training, validation, and production serving—regardless of whether the host environment is Python, Rust, or C++.
- **⚡ Native Rust Performance**: Engineered in Rust with zero-copy tensor slicing, multi-core work-stealing parallelism (Rayon), and SIMD-friendly layout transformations. Get native throughput even when invoked from Python.
- **📄 Human-Readable & Transparent**: The declarative `.morf` dataflow format is intuitive and self-explanatory. Unlike opaque computational graphs (such as TorchScript or ONNX runtime graphs), `.morf` pipelines can be reviewed, diffed, and understood across multidisciplinary teams in seconds.
- **📦 Lean, Modular Production Deployments**: Dynamic action plugins (`cdylib`) ensure that production containers load only the specific processing kernels required by active pipelines, dramatically reducing container image sizes and security attack surfaces.
- **🚀 Decoupled Pipeline Shipping**: Update, test, and ship new data pipelines and transformations alongside AI model artifacts without rebuilding or redeploying server binaries.

---

## ⚡ 30-Second Quickstart

### 1. Define your pipeline (`pipeline.morf`)
```morf
accept $input

$input
    >> to_tensor(color="rgb", normalize=true)
    >> resize(width=512, height=512, filter="bilinear")
    >> color_adjust(contrast=1.15, saturation=1.05)
    >> gaussian_blur(sigma=1.0)
    >> to_image
    >> emit
```

### 2. Run in Python (Zero-Copy NumPy)
```python
import morflow
import numpy as np

# Load pipeline and execute with zero memory overhead
pipeline = morflow.load("pipeline.morf")
output_array = pipeline.run(input_numpy_array)
```

### 3. Or run in Rust
```rust
use pipeline::{Morflow, Payload};

let mut pipeline = Morflow::load("pipeline.morf")?;
let output = pipeline.run(Payload::Image(image))?.into_single()?;
```

---

## 🛠️ Installation & Building

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (edition 2021+)
- Python 3.8+ (optional, for Python bindings)

### Build Everything (Engine & Action Plugins)
```bash
./build-release.sh
```
This compiles the engine and builds action plugins into `target/release/actions/`.

### Install Python Bindings
```bash
cd bindings/python
maturin develop --release
```

---

## 🧩 Extending Morflow: Custom Actions

Custom processing actions are written as lightweight Rust shared libraries:

```rust
// actions/custom_kernel/src/lib.rs
use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType { DataType::Tensor }

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType { DataType::Tensor }

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    match payload.unwrap_payload() {
        Payload::Tensor(mut tensor) => {
            // High-performance in-place tensor transformation
            Payload::Tensor(tensor)
        }
        other => other.clone(),
    }
}
```
Compile to `.so`/`.dll` and drop it into your actions directory—Morflow discovers and registers it dynamically at runtime with zero host recompilation.

---

## 📂 Examples

Explore end-to-end host examples in both Rust and Python under [`examples/`](examples/):
- **Image Processing**: Color grading, spatial resampling, and spatial filtering.
- **Audio DSP Mastering**: Dynamic range compression, biquad EQ, multi-channel stereo widening, and peak limiting.
- **Audio Split**: Parallel stereo channel extraction with multiple named stream outputs.


---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
