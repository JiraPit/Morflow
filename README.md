# Morflow

**Morflow** is a data processing engine that allows developers to define processing pipelines as human-readable `.morf` files.

Instead of hardcoding data transformations into application code or manually copying training scripts into production backends, Morflow provides a portable, high-performance runtime that executes the exact same `.morf` pipeline anywhere.

---

## 🎯 Why Morflow?

Morflow was created to solve core challenges in modern data and AI engineering:

- **🔒 Guaranteed Identical Operations (Solving Train-Serve & Cross-Language Skew)**: In typical AI workflows, preprocessing code written in Python training notebooks is copy-pasted or rewritten in Rust, JavaScript, Java, or C++ for production services. This rewrite introduces subtle numerical discrepancies, library differences, and critical bugs (**train-serve and cross-language skew**). With Morflow, you write the pipeline once in a `.morf` file and execute the exact same deterministic operations across all host languages (Python, Rust, Java, JS) in both training and production serving.
- **⚡ High Performance Regardless of Host Language**: Written in Rust, Morflow provides zero-copy memory operations, automatic flow parallelism, and multi-threaded CPU execution. When calling Morflow from Python or another host language, you get the full speed and multi-core scalability of native Rust.
- **📄 Developer-Friendly & Self-Explanatory Pipelines**: Graph formats like TorchScript or ONNX are difficult to inspect, debug, or understand without specialized tools. A `.morf` file is clean, declarative, and easily understood by developers across different teams (data science, backend, infra) at a glance.
- **📦 Lean Production Images**: Rather than bundling massive, monolithic dependencies, Morflow ensures production deployments include only the specific actions required by the active pipelines, drastically reducing container image sizes.
- **🧩 Rich "Action Packs" & Effortless Custom Actions**: Designed around an ecosystem of modular, reusable **Action Packs** (for computer vision, audio DSP, multimodal preprocessing, etc.) alongside a lightweight interface that makes creating custom actions straightforward and fast.
- **🚀 Ship AI & Data Pipelines Together Without Server Rebuilds**: Because `.morf` pipelines and actions are decoupled from host application logic, data processing pipelines can be updated and shipped alongside AI models without requiring server code recompilation or backend service rebuilds.

---

## ⚡ Quickstart

### 1. Install Morflow

```bash
# Python
pip install morflow

# Node.js
npm install morflow

# Rust
cargo add morflow
```

### 2. Define your pipeline (`pipeline.morf`)

```morf
import base.latest
import image_essentials.latest

accept $input

$input
    >> to_tensor(color="rgb", normalize=true)
    >> resize(width=512, height=512, filter="bilinear")
    >> color_adjust(contrast=1.15, saturation=1.05)
    >> gaussian_blur(sigma=1.0)
    >> to_image
    >> emit
```

### 3. Prepare Actions

```bash
morflow prep pipeline.morf
```

### 4. Run in your application

**Python (Zero-Copy NumPy)**:
```python
import morflow
import numpy as np

# Load pipeline and execute with zero memory overhead
pipeline = morflow.load("pipeline.morf")
output_array = pipeline.run(input_numpy_array)
```

**JavaScript / TypeScript (Node.js)**:
```javascript
import morflow from 'morflow';

// Load pipeline and execute asynchronously without blocking the event loop
const pipeline = morflow.load('pipeline.morf');
const outputTensor = await pipeline.run(inputTypedArray);
```

Full end-to-end examples across Rust, Python, and JavaScript are available in the [`examples/`](examples) directory.

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

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
