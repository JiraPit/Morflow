# Morflow

**Morflow** is a data processing engine that allows developers to define processing pipelines as human-readable `.morf` files.

Instead of hardcoding data transformations into application code or manually copying training scripts into production backends, Morflow provides a portable, high-performance runtime that executes the exact same `.morf` pipeline anywhere.

---

## 🎯 Why Morflow?

Morflow was created to solve core challenges in modern data and AI engineering:

- **🔒 Guaranteed Identical Operations (Solving Train-Serve Skew)**: In typical AI workflows, preprocessing code written in Python training notebooks is copy-pasted or rewritten for production services. This introduces subtle discrepancies (**train-serve skew**) and maintenance bugs. With Morflow, you define the pipeline once in a `.morf` file and run it identically in both training and production serving across multiple languages.
- **⚡ High Performance Regardless of Host Language**: Written in Rust, Morflow provides zero-copy memory operations, automatic flow parallelism, and multi-threaded CPU execution. When calling Morflow from Python or another host language, you get the full speed and multi-core scalability of native Rust.
- **📄 Developer-Friendly & Self-Explanatory Pipelines**: Graph formats like TorchScript or ONNX are difficult to inspect, debug, or understand without specialized tools. A `.morf` file is clean, declarative, and easily understood by developers across different teams (data science, backend, infra) at a glance.
- **📦 Lean Production Images**: Rather than bundling massive, monolithic dependencies, Morflow ensures production deployments include only the specific actions required by the active pipelines, drastically reducing container image sizes.
- **🧩 Rich "Action Packs" & Effortless Custom Actions**: Designed around an ecosystem of modular, reusable **Action Packs** (for computer vision, audio DSP, multimodal preprocessing, etc.) alongside a lightweight interface that makes creating custom actions straightforward and fast.
- **🚀 Ship AI & Data Pipelines Together Without Server Rebuilds**: Because `.morf` pipelines and actions are decoupled from host application logic, data processing pipelines can be updated and shipped alongside AI models without requiring server code recompilation or backend service rebuilds.

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
