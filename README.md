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
    <version>0.2.1</version>
</dependency>
```

### 2. Define your pipeline (`pipeline.morf`)

```morf
import base.latest
from base.latest import to_tensor
from image_basics.latest import resize, color_adjust, gaussian_blur, to_image

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

Shape checks use native action contracts, with execution validating known output dimensions. See [shape documentation](https://morflow.org/doc#types-shapechecking) for coverage, unresolved metadata, and local validation.

### Select Composite outputs

QR returns a Composite containing Q and R, in that order:

```morf
import linalg_basics/latest
accept Tensor[3,2] $matrix

$matrix >> qr >> $parts
$parts[0] >> emit("q")
$parts[1] >> emit("r")
```

Integer indexes select one payload while preserving its type and dimensions. Indexes start at zero. You can chain selection and tensor indexing, such as `$parts[0][0]` for the first row of Q. QR's component contract lets `check` track Q as Tensor[3,2] and R as Tensor[2,2].

Composite inputs can declare an ordered list of component shapes, including nested lists:

```morf
import linalg_basics/latest
accept Composite[Tensor[2,3], Tensor[3,4]] $matrices
$matrices >> matmul >> emit("product")
```

The first matrix is the left operand and the second is the right operand; this produces Tensor[2,4]. The checker and runtime use the same ordered shape contract. QR's `$parts` can also flow directly into `matmul` to reconstruct the original matrix.

### Action versions and cache

Imports select the action version used by both preparation and execution:

```morf
from image_basics/0.2.0 import resize
from audio_basics/latest import gain
```

Exact versions are stored in filenames such as `resize_action-0.2.0-linux-x86_64.so`. A `latest` import uses a separate file such as `gain_action-latest-linux-x86_64.so`. Each `morflow prep` or `morflow install` resolves the newest stable release again, compares published SHA-256 checksums with the cached bytes, and refreshes `latest` when needed. Exact versions remain installed alongside it.

Each binary has a checksum receipt recording its concrete release and provenance. Loading a pipeline verifies the receipt and binary locally; execution uses the loaded actions until the pipeline is reloaded. File and string loading APIs use the same cache, without pipeline lock files or runtime downloads.

Actions resolve through imports or an explicit `pack/version/action` call. Use aliases to select multiple versions of a pack:

```morf
import image_basics/0.1.0 as old_image
import image_basics/0.2.0 as new_image
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

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

Native shape contracts use `Dimension::Known(usize)` and `Dimension::Unknown`. This retains partially known shapes such as `Tensor[*,*,3]` and distinguishes real zero-length dimensions from wildcards. Tensor `each` preserves row dimensions: `Tensor[*,3]` produces `Tensor[3]` loop variables. See [shape documentation](https://morflow.org/doc#types-shapechecking).
