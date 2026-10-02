# OpenBLAS action packs

These packs use a shared OpenBLAS library installed on the machine running the pipeline. Action binaries contain only the Rust wrapper and Morflow types. They do not embed OpenBLAS or link to it at build time.

| Pack | Actions | OpenBLAS operations |
| --- | --- | --- |
| `linalg_blas/0.1.0` | `matmul`, `dot`, `outer` | SGEMM, SDOT, SGER |
| `linalg_blas/0.1.0` | `inv`, `det`, `qr`, `cholesky` | LAPACKE LU, inversion, Householder QR, Cholesky |
| `tensor_blas/0.1.0` | `concat`, `repeat`, `roll` | Floating-point block copies with SCOPY and DCOPY |

Use `tensor_basics` for reshape, transpose, permute, squeeze, unsqueeze, flatten, and cast. Those operations either create tensor views or have no corresponding BLAS operation.

## Using the packs

```morf
from linalg_blas/0.1.0 import matmul
from tensor_basics/0.3.1 import transpose
accept Composite[Tensor[2,3], Tensor[3,4]] $matrices

$matrices >> matmul >> transpose >> emit("transposed_product")
```

`morflow prep` downloads action binaries as usual. `morflow check` and pipeline loading use the same type and shape rules as the corresponding basics actions, with additional checks for the numerical backend's LP64 integer limits. Neither step loads OpenBLAS. Each action checks the runtime dependency inside `process`, after the engine's runtime shapecheck.

The first successful execution loads OpenBLAS and retains its handle and function pointers. Subsequent calls reuse them. A missing library or required symbol produces `Payload::Error`; there is no substitution with a basics action. A failed library load is not cached, so a later run can retry after installation or configuration.

## Installing the shared library

Install an **LP64 OpenBLAS** build through your operating system's package manager. LP64 uses 32-bit BLAS dimensions and is the usual `libopenblas` interface. ILP64 builds are rejected to prevent calling functions with the wrong integer ABI. Factorization actions require an OpenBLAS build exporting the LAPACKE functions listed above. A build without LAPACKE can still execute the CBLAS actions.

Default library names are `libopenblas.so.0`/`libopenblas.so` on Linux, `libopenblas.dylib` on macOS, and `libopenblas.dll`/`openblas.dll` on Windows. Homebrew's standard OpenBLAS paths are also searched on macOS. The platform loader resolves OpenBLAS's own shared dependencies.

Select a specific installation before the first execution:

```bash
export MORFLOW_OPENBLAS_LIBRARY=/path/to/libopenblas.so
```

An explicitly configured path is authoritative: if it fails, another installation is not silently selected. Once successfully loaded, changing the environment variable does not replace that process's backend. Restart the process to change installations.

## Numerical behavior

The linear algebra actions preserve the basics interfaces, argument order, output kinds, shapes, and batch broadcasting rules. As in `linalg_basics`, calculations and results use F32; other numerical input dtypes are converted to F32. Different reduction and factorization algorithms can produce small floating-point differences.

QR uses Householder factorization. It returns `[Q, R]`, with Q shaped `[m,n]` and R shaped `[n,n]`. For wide input matrices, the final `n−m` columns of Q and rows of R are zero. Householder QR can produce different column signs from the basics implementation; Q × R reconstructs the input. Singular determinants return zero, singular inverses fail, and Cholesky fails for matrices that are not positive definite. All algorithms handle empty dimensions.

CBLAS dimensions and leading dimensions must fit signed 32-bit integers. Shapecheck rejects known dimensions beyond that limit without loading OpenBLAS; dimensions that remain unknown are checked when executing.

## Performance and threading

Contiguous F32 operands and dense matrix transpose views are passed directly to SGEMM without making input copies. Other views and dtypes are materialized or converted only when needed. Inversion uses the writable output buffer for LU and inversion; batched determinants reuse their matrix and pivot buffers. Repeat fills its final output directly rather than allocating a larger intermediate tensor for each repeated axis.

Tensor operations preserve every supported dtype and copy sliced roll inputs directly into the final output without intermediate tensors. Floating-point blocks use OpenBLAS copy routines where useful; small blocks, integer data, and unaligned storage use direct memory copies. Copy performance depends on tensor size, axis, memory bandwidth, and the installed OpenBLAS build. These packs do not promise that every small operation is faster than basics.

The engine schedules independent flows. OpenBLAS may also use its own workers. Configure its thread count **before starting the process**. For many concurrent flows, start with:

```bash
OPENBLAS_NUM_THREADS=1 python your_program.py
```

For a pipeline dominated by one large matrix multiplication, compare larger thread counts. The actions do not change OpenBLAS's global thread settings during execution, avoiding races between flows. Single-threaded OpenBLAS builds cannot gain internal parallelism from this variable.

## Building locally

Building does not require OpenBLAS headers, a static archive, or an installed shared library:

```bash
bash scripts/build-actions.sh linalg_blas
bash scripts/build-actions.sh tensor_blas
```

Cargo package names are unique (`linalg_blas_matmul`, for example). Public action names remain `matmul`, recorded by `package.metadata.morflow.action`. Local packaging and release workflows preserve those public names and generate the usual versioned binaries, receipts, and catalogs.

## Validation and performance measurements

Action tests require a shared OpenBLAS installation to exercise numerical execution. They also start isolated processes with a missing library path to verify that shapecheck still succeeds and `process` reports the dependency error.

Two local measurement tools compare execution against basics; inputs and validation plans are created once and reused:

```bash
cargo run --release -p morflow-openblas --example matmul_benchmark
cargo run --release -p morflow-openblas --example copy_benchmark
```

These measurements exclude pipeline loading and measure steady-state execution. They are implementation checks, not a portable speed guarantee. The benchmark prints the actual timings and speed ratio for each size.

Interface reference: [OpenBLAS CBLAS declarations](https://github.com/OpenMathLib/OpenBLAS/blob/develop/cblas.h) and [LAPACKE declarations](https://github.com/OpenMathLib/OpenBLAS/blob/develop/lapack-netlib/LAPACKE/include/lapacke.h).
