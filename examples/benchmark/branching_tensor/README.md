# Branching tensor benchmark

Compare **NumPy**, **morf-basic**, and **morf-blas** on the same fixed graph: a 512×512 float32 input, 16 branches, 49 flows, 114 native action calls, and 32 named outputs.

A shared producer scales and offsets the input. Independent branches roll, multiply, add, and clamp it. Each branch feeds two downstream flows: one transposes and rolls its matrix; the other sums its rows.

- [`morf-basic.morf`](morf-basic.morf) uses the existing basics actions.
- [`morf-blas.morf`](morf-blas.morf) replaces `roll` with `tensor_blas/0.1.0`. Transpose, arithmetic, and reductions keep their existing actions. This graph does not contain linear algebra operations, so it does not use `linalg_blas`.

The two Morflow graphs perform identical operations in the same order. This comparison measures the OpenBLAS-backed roll implementation within the existing many-flow workload.

## Setup

From the repository root:

```sh
python3 -m pip install -r examples/benchmark/branching_tensor/requirements.txt
```

Install the current Morflow Python binding and prepare the actions. See [binding and local action setup](../image/README.md#setup). For local builds, also build the new pack:

```sh
bash scripts/build-actions.sh tensor_blas
```

When the actions are published, prepare both pipelines with:

```sh
morflow prep examples/benchmark/branching_tensor/morf-basic.morf
morflow prep examples/benchmark/branching_tensor/morf-blas.morf
```

`morf-blas` execution requires a shared LP64 OpenBLAS library. See [OpenBLAS setup](../../../backends/openblas/README.md). Set `MORFLOW_OPENBLAS_LIBRARY` before starting the benchmark to select a specific installation.

## Run

```sh
# Compare all three backends using locally prepared actions
OPENBLAS_NUM_THREADS=1 python3 examples/benchmark/branching_tensor/benchmark.py \
  --numpy --morf-basic --morf-blas \
  --numpy-workers 4 --threads 4 \
  --actions-path target/release/actions --output /tmp/tensor-benchmark.json

# Compare the two Morflow implementations
OPENBLAS_NUM_THREADS=1 python3 examples/benchmark/branching_tensor/benchmark.py \
  --morf-basic --morf-blas --actions-path target/release/actions

# Run one backend alone
python3 examples/benchmark/branching_tensor/benchmark.py --numpy
python3 examples/benchmark/branching_tensor/benchmark.py --morf-basic \
  --actions-path target/release/actions
OPENBLAS_NUM_THREADS=1 python3 examples/benchmark/branching_tensor/benchmark.py --morf-blas \
  --actions-path target/release/actions
```

With no backend flags, all three run. Defaults are 10 graph executions per sample, five samples, three warmup executions, and seed 0. Override these with `--iterations`, `--samples`, `--warmup`, and `--seed`. Dimensions and branch count are fixed.

`--numpy-workers 1` runs NumPy branches sequentially. Higher values use a persistent thread pool with one task per branch, whose two consumers run sequentially. Both Morflow variants schedule the consumers as separate flows. `--threads` sets the Rayon worker count before loading.

Set `OPENBLAS_NUM_THREADS` before launching Python. NumPy may also use OpenBLAS, and each native library can manage its own threads. These settings do not guarantee identical CPU usage.

## Load once, run repeatedly

The input tensor is generated once in memory. Each selected Morflow backend calls `morflow.load()` on its own `.morf` file **once**, then reuses that pipeline and its wrapped input for validation, warmup, and every timed `run()`. Runs execute sequentially; parallelism occurs within each run.

Pipeline loading, input generation, correctness checks, warmup, and thread-pool creation are outside timing. Timing includes full graph execution, scheduling, output materialization, and output release. Backend order reverses on alternate samples to reduce ordering bias.

## Validation and results

Every selected backend must match the same NumPy reference before timing. All output names, shapes, and float32 dtypes must match. Matrices use `atol=1e-6, rtol=0`; row sums use `atol=1e-3, rtol=1e-5` to allow different accumulation orders. Validation failures do not write results.

JSON files default to `tensor/results/tensor-<UTC timestamp>.json`, or the supplied `--output` path. Schema version 2 identifies results as `numpy`, `morf-basic`, and `morf-blas`, and records both pipeline sources, raw sample timings, timing statistics, throughput, validation errors, and thread/library environment settings.

Reports include a time ratio for every selected backend pair. For example, `morf_basic_time_divided_by_morf_blas_time` above 1 means `morf-blas` was faster in that run.

## Tests

```sh
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  python3 -m unittest discover -s examples/benchmark/branching_tensor -t examples/benchmark
```

Tests verify one load per selected pipeline across repeated runs, identical graph bodies, backend selection, NumPy threading, JSON metadata, native output agreement, and input preservation.
