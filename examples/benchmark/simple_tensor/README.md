# Sequential tensor benchmark

A single 256 × 256 float32 tensor passes through one long dependency chain: twelve matrix inversions, each followed by a transpose, then a Cholesky decomposition. These are deliberately expensive numerical operations. There are 25 native action calls and one output. This synthetic workload measures sequential numerical processing; `branching_tensor` measures independent branch scheduling.

The input is a reproducible, well-conditioned positive definite matrix. Each inverse/transpose pair feeds the next pair. Twelve inversions return approximately the original matrix, allowing the final Cholesky factor to be checked independently. Floating-point differences between implementations are expected; output validation uses absolute tolerance 2e-5 and relative tolerance 2e-4.

## Backends

- `--numpy`: NumPy matrix inversion and Cholesky.
- `--morf-basic`: `linalg_basics` inversion and Cholesky.
- `--morf-blas`: `linalg_blas` inversion and Cholesky, using shared OpenBLAS with LAPACKE symbols.

Both Morflow pipelines use `tensor_basics` for transpose. No flags runs all three backends; flags select any subset. This benchmark has no NumPy branch worker pool. Numerical libraries may use their own threads.

## Run

From the repository root, install the dependencies and the current Morflow Python binding:

```sh
python3 -m pip install -r examples/benchmark/simple_tensor/requirements.txt
morflow prep examples/benchmark/simple_tensor/morf-basic.morf
morflow prep examples/benchmark/simple_tensor/morf-blas.morf
OPENBLAS_NUM_THREADS=1 python3 examples/benchmark/simple_tensor/benchmark.py
```

For locally built, prepared actions:

```sh
OPENBLAS_NUM_THREADS=1 python3 examples/benchmark/simple_tensor/benchmark.py \
  --numpy --morf-basic --morf-blas --actions-path target/release/actions
```

The BLAS backend needs a shared LP64 OpenBLAS library with LAPACKE available at execution time. Set `MORFLOW_OPENBLAS_LIBRARY` to its path when automatic discovery is insufficient. Set numerical library thread variables before starting Python; `--threads` controls Morflow's Rayon workers. Record and use comparable thread settings when comparing results.

## Measurement and results

Each selected Morflow pipeline is loaded exactly once with `morflow.load`; its input wrapper is created once. Input generation, reference calculation, validation, warmup and pipeline loading happen outside timing. Every measured call reuses the same input from memory. Timing includes execution, output conversion, materialization and release.

Use `--iterations`, `--samples`, `--warmup`, and `--seed` to control the run. Reports include pipeline sources, workload dimensions, environment, validation errors, per-backend timing statistics and pairwise timing ratios. JSON is saved under `simple_tensor/results/`; `--output` selects another file. Generated reports and artifacts are ignored by the benchmark `.gitignore`.

```sh
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" OPENBLAS_NUM_THREADS=1 \
  python3 -m unittest discover -s examples/benchmark/simple_tensor -t examples/benchmark
```

The BLAS pipeline declares `plugin openblas/0.1.1`. Prepare it with `morflow prep`, or build locally with `bash scripts/build-plugins.sh openblas` and set `MORFLOW_PLUGINS_PATH="$PWD/target/release/plugins"`. Shared system OpenBLAS is needed for execution.
