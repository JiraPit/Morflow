# Tensor transformation benchmark

Compare NumPy and Morflow on the fixed graph in [`pipeline.morf`](pipeline.morf): a 512×512 float32 input, 16 branches, 49 flows, 114 native action calls, and 32 named outputs.

A shared producer scales and offsets the input. Independent branches roll, multiply, add, and clamp it. Each branch feeds two downstream flows: one transposes and rolls its matrix; the other sums its rows. Morflow schedules ready flows according to these dependencies.

## Setup

From the repository root:

```sh
python3 -m pip install -r examples/benchmark/tensor/requirements.txt
```

For Morflow, install the current Python binding and prepare the actions. See [binding and local action setup](../image/README.md#setup), or prepare published actions directly:

```sh
morflow prep examples/benchmark/tensor/pipeline.morf
```

## Run

```sh
# Compare both backends using locally prepared actions
python3 examples/benchmark/tensor/benchmark.py --numpy --morflow \
  --numpy-workers 4 --threads 4 \
  --actions-path target/release/actions --output /tmp/tensor-benchmark.json

# Run either backend alone
python3 examples/benchmark/tensor/benchmark.py --numpy
python3 examples/benchmark/tensor/benchmark.py --morflow \
  --actions-path target/release/actions
```

Defaults are 10 graph executions per sample, five samples, three warmup executions, and seed 0. Override these with `--iterations`, `--samples`, `--warmup`, and `--seed`. Dimensions and branch count are fixed; there are no graph-generation options.

`--numpy-workers 1` runs branches sequentially. Higher values use a persistent thread pool with one task per branch, whose two consumers run sequentially. Morflow schedules the consumers as separate flows. `--threads` sets its Rayon worker count before loading. NumPy may also use library-managed threads; these settings do not guarantee identical CPU usage.

## Load once, run repeatedly

The input tensor is generated once in memory. Morflow calls `load()` on `pipeline.morf` **once**, then reuses that instance for validation, warmup, and every timed `run()`. Each execution starts from the original tensor. Pipeline loading, input generation, correctness checks, warmup, and thread-pool creation are outside timing.

Timing includes full graph execution, Python/native input conversion, scheduling, all output materialization, and output release. This synthetic workload compares complete implementations; it does not isolate scheduler overhead.

## Validation and results

All output names, shapes, and float32 dtypes must match NumPy before timing. Matrices use `atol=1e-6, rtol=0`; row sums use `atol=1e-3, rtol=1e-5` to allow different floating-point accumulation orders. Validation failures do not write results.

JSON files default to `tensor/results/tensor-<UTC timestamp>.json`, or the path supplied by `--output`. Reports include raw sample durations, mean/median/min/max and standard deviation of sample means, graphs per second, environment and thread settings, output errors, and the loaded pipeline source. With both backends, NumPy time divided by Morflow time is also reported; above 1 means Morflow was faster in that run.

## Tests

```sh
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  python3 -m unittest discover -s examples/benchmark/tensor -t examples/benchmark
```

Tests cover one pipeline load across repeated runs, sequential/parallel NumPy agreement, JSON metadata, native output agreement, repeated execution, and input preservation.
