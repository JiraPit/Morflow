# Benchmarks

Each benchmark has its own runner, pipeline, tests, dependencies, and generated results:

| Benchmark | Backends | Instructions |
| --- | --- | --- |
| Image transformation | PIL/NumPy, Morflow | [image/README.md](image/README.md) |
| Branching tensor transformation | NumPy, morf-basic, morf-blas | [branching_tensor/README.md](branching_tensor/README.md) |
| Sequential tensor transformation | NumPy, morf-basic, morf-blas | [simple_tensor/README.md](simple_tensor/README.md) |

Shared argument validation, timing statistics, and JSON writing live in `common.py`.
All benchmarks load their inputs and Morflow pipelines once, then repeatedly execute from memory.

From the repository root:

```sh
python3 examples/benchmark/image/benchmark.py --pil
python3 examples/benchmark/branching_tensor/benchmark.py --numpy
python3 examples/benchmark/simple_tensor/benchmark.py --numpy

# Run all tests, including native integration with a prepared cache.
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  python3 -m unittest discover -s examples/benchmark -t examples/benchmark -p 'test_*.py'
```

Results are saved in each benchmark's `results/` folder unless `--output` is supplied.
