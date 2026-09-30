<p align="center">
  <a href="https://morflow.org">
    <img src="assets/logo.svg" alt="Morflow" width="300" />
  </a>
</p>

<p align="center">
  <strong>Modular, high-performance data processing engine built for AI/ML input/output processing and tensor transformations.</strong>
</p>

<p align="center">
  <a href="https://morflow.org"><strong>🌐 Website (morflow.org)</strong></a> •
  <a href="https://morflow.org/doc"><strong>📖 Documentation</strong></a> •
  <a href="examples"><strong>💡 Examples</strong></a> •
  <a href="LICENSE"><strong>📄 License</strong></a>
</p>

---

**Morflow** (pronounced *morph-flow*) is a modular, high-performance data processing engine built for AI/ML input/output processing and tensor transformations. Define processing pipelines for images, audio, tensors and more in declarative `.morf` files and execute them identically across training and production serving with zero-copy, multi-threaded performance and zero train-serve skew.

Instead of hardcoding data transformations into application code or manually copying training scripts into production backends, Morflow provides a portable, high-performance runtime that executes the exact same deterministic operations anywhere.

Visit the official website and documentation at **[morflow.org](https://morflow.org)**.

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

# Java (Maven)
<dependency>
    <groupId>org.morflow</groupId>
    <artifactId>morflow</artifactId>
    <version>0.2.0</version>
</dependency>
```

### 2. Define your pipeline (`pipeline.morf`)

```morf
import base.latest
from base.latest import to_tensor
from image_essentials.latest import resize, color_adjust, gaussian_blur, to_image

accept Image $img_in
accept IntArg $target_width = 512
accept IntArg $target_height = 512

$img_in
    >> to_tensor(color="rgb", dtype="f32", layout="hwc", normalize=true)
    >> resize(width=$target_width, height=$target_height, filter="bilinear")
    >> color_adjust(contrast=1.15, saturation=1.05, brightness=0.02)
    >> gaussian_blur(sigma=1.2)
    >> to_image(color="rgba", dtype="u8")
    >> emit
```

### 3. Prepare Actions

Pre-downloads all actions required by your `.morf` file ahead of time, ensuring the runtime executes completely offline with zero dynamic downloads:

```bash
morflow prep pipeline.morf
```

Shape checks use native action contracts, with execution validating known output dimensions. See [shape contracts](docs/shape-contracts.md) for coverage, unresolved metadata, and local validation.

### Action versions and cache

Imports select the action version used by both preparation and execution:

```morf
from image_essentials/0.2.0 import resize
from audio_essentials/latest import gain
```

Exact versions are stored in filenames such as `resize_action-0.2.0-linux-x86_64.so`. A `latest` import uses a separate file such as `gain_action-latest-linux-x86_64.so`. Each `morflow prep` or `morflow install` resolves the newest stable release again, compares published SHA-256 checksums with the cached bytes, and refreshes `latest` when needed. Exact versions remain installed alongside it.

Each binary has a checksum receipt recording its concrete release and provenance. Loading a pipeline verifies the receipt and binary locally; execution uses the loaded actions until the pipeline is reloaded. File and string loading APIs use the same cache, without pipeline lock files or runtime downloads.

Actions resolve through imports or an explicit `pack/version/action` call. Use aliases to select multiple versions of a pack:

```morf
import image_essentials/0.1.0 as old_image
import image_essentials/0.2.0 as new_image
```

Qualified calls such as `old_image.resize(...)` and `new_image.resize(...)` select their respective versions. `morflow list` displays installed versions and the concrete release behind each `latest` entry. Set `MORFLOW_ACTIONS_PATH` to use a prepared custom cache.

### 4. Run in your application

**Python**:
```python
import morflow
import numpy as np

# Load pipeline and execute with zero memory overhead
pipeline = morflow.load("pipeline.morf")
# Payload type is always explicit. A bare array is a plain tensor; wrap it to
# send it as an image.
output_array = pipeline.run(morflow.Image(input_numpy_array, color="rgb"))
```

**JavaScript / TypeScript (Node.js)**:
```javascript
import morflow from 'morflow';

// Load pipeline and execute asynchronously without blocking the event loop
const pipeline = morflow.load('pipeline.morf');
// Payload type is always explicit. Without payloadType this is a plain tensor.
const outputTensor = await pipeline.run({
  data: inputTypedArray,
  shape: [height, width, 3],
  dtype: 'u8',
  payloadType: 'image',
  colorSpace: 'rgb'
});
```

**Java**:
```java
import org.morflow.*;

// Load pipeline and execute with direct ByteBuffer zero-copy support
try (Pipeline pipeline = Morflow.load("pipeline.morf")) {
    MorflowTensor output = pipeline.run(inputTensor.asImage("rgb"));
}
```

**Rust**:
```rust
use morflow::{ColorSpace, Image, Morflow, Payload};

// Load pipeline and execute natively
let mut pipeline = Morflow::load("pipeline.morf")?;
let outputs = pipeline.run(Payload::Image(input_image))?;
```

Full end-to-end examples across Rust, Python, JavaScript, and Java are available in the [`examples/`](examples) directory, and comprehensive API guides are available in the **[Official Documentation (morflow.org/doc)](https://morflow.org/doc)**.

---

## 🧩 Extending Morflow: Custom Actions

Custom processing actions are written as lightweight Rust shared libraries:

```rust
// actions/custom_kernel/src/lib.rs
use core_types::{ActionArgs, DataType, Payload, Shape, ShapeResult};

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

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, _args: ActionArgs) -> ShapeResult {
    // Pass-through: the output has the same shape as the input.
    input.into()
}
```
Compile the action as a `.so`/`.dll`/`.dylib`, then package it with its Cargo version and checksum receipt:

```bash
python3 scripts/package_action.py actions/custom_pack/custom_kernel/Cargo.toml \
    target/release/libcustom_kernel.so --cache ./actions_cache
```

Import it with `from custom_pack/0.1.0 import custom_kernel` and point `MORFLOW_ACTIONS_PATH` at the cache. `scripts/build-actions.sh` packages repository actions in the same format. Existing unversioned cache files are excluded from version-aware resolution; run `morflow prep` to prepare published actions.

For local test fixtures, build with `cargo build --workspace` and run `python3 scripts/prepare_test_actions.py`. This explicitly creates development `latest` aliases under `target/debug/actions`; published `latest` aliases are refreshed by `morflow prep` or `morflow install`.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
