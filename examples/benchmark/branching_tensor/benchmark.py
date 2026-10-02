#!/usr/bin/env python3
"""Stress a many-flow tensor graph with NumPy, morf-basic, or morf-blas; save validated JSON timings."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# Direct script execution needs the benchmark root to import shared helpers.
sys.path.insert(0, str(ROOT.parent))

from common import nonnegative, positive, save_json, summary  # noqa: E402

INPUT_SHAPE = (512, 512)
BRANCHES = 16
PIPELINES = {name: ROOT / f"{name}.morf" for name in ("morf-basic", "morf-blas")}


def numpy_runner(tensor: np.ndarray, branches: int, executor=None):
    def branch(shared, index):
        value = np.roll(shared, index + 1, axis=1)
        value = value * np.float32(1 + (index % 8) / 8)
        value = value + np.float32((index % 5) / 16)
        value = np.clip(value, np.float32(-1), np.float32(1))
        matrix = np.ascontiguousarray(np.roll(value.T, index + 1, axis=0))
        rows = np.sum(value, axis=1, dtype=np.float32)
        return index, matrix, rows

    def run():
        shared = tensor * np.float32(0.5) + np.float32(0.25)
        if executor is None:
            values = [branch(shared, index) for index in range(branches)]
        else:
            futures = [
                executor.submit(branch, shared, index) for index in range(branches)
            ]
            values = [future.result() for future in futures]
        result = {}
        for index, matrix, rows in values:
            result[f"matrix_{index}"] = matrix
            result[f"rows_{index}"] = rows
        return result

    return run


def morflow_runner(tensor, backend):
    try:
        import morflow

        # Setup once; all validation, warmup, and measured calls reuse this instance.
        pipeline = morflow.load(str(PIPELINES[backend]))
        wrapped = morflow.Tensor(tensor)
    except (ImportError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Install the current Morflow Python binding and prepare the pipeline actions. "
            + str(error)
        ) from error
    return lambda: pipeline.run(wrapped)


def validate(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise RuntimeError(
            "Output names do not match the graph. Rebuild the Python binding if it is stale."
        )
    errors = {}
    for name, reference in expected.items():
        value = np.asarray(actual[name])
        if value.shape != reference.shape or value.dtype != np.float32:
            raise RuntimeError(
                f"{name}: expected {reference.shape}, float32; got {value.shape}, {value.dtype}"
            )
        # Reduction accumulation order can differ; transformed matrices use tighter bounds.
        atol, rtol = (1e-3, 1e-5) if name.startswith("rows_") else (1e-6, 0)
        np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol, err_msg=name)
        errors[name] = float(np.max(np.abs(value - reference)))
    return {
        "passed": True,
        "max_absolute_error_by_output": errors,
        "matrix_tolerance": {"atol": 1e-6, "rtol": 0},
        "reduction_tolerance": {"atol": 1e-3, "rtol": 1e-5},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy", action="store_true")
    parser.add_argument("--morf-basic", action="store_true")
    parser.add_argument("--morf-blas", action="store_true")
    parser.add_argument(
        "--numpy-workers",
        type=positive,
        default=1,
        help="1: sequential NumPy; >1: persistent thread pool",
    )
    parser.add_argument("--threads", type=positive, help="Morflow Rayon worker count")
    parser.add_argument("--iterations", type=positive, default=10)
    parser.add_argument("--samples", type=positive, default=5)
    parser.add_argument("--warmup", type=nonnegative, default=3)
    parser.add_argument("--seed", type=nonnegative, default=0)
    parser.add_argument("--actions-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    available = ("numpy", "morf-basic", "morf-blas")
    names = [
        name for name in available if getattr(args, name.replace("-", "_"))
    ] or list(available)
    if args.threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    if args.actions_path:
        os.environ["MORFLOW_ACTIONS_PATH"] = str(args.actions_path.resolve())
    executor = None
    try:
        started = datetime.now(timezone.utc)
        sources = {name: path.read_text() for name, path in PIPELINES.items()}
        # Generate input once. Every graph execution starts from this in-memory tensor.
        tensor = (
            np.random.default_rng(args.seed)
            .uniform(-2, 2, INPUT_SHAPE)
            .astype(np.float32)
        )
        reference = numpy_runner(tensor, BRANCHES)()
        if args.numpy_workers > 1 and "numpy" in names:
            executor = ThreadPoolExecutor(max_workers=args.numpy_workers)
        runners = {
            name: (
                numpy_runner(tensor, BRANCHES, executor)
                if name == "numpy"
                else morflow_runner(tensor, name)
            )
            for name in names
        }
        checks = {}
        for name, run in runners.items():
            checks[name] = validate(run(), reference)
            for _ in range(args.warmup):
                run()
        del reference
        # Repeated execution only: runners never load pipelines or regenerate inputs.
        durations = {name: [] for name in names}
        for sample in range(args.samples):
            for name in names if sample % 2 == 0 else reversed(names):
                begin = time.perf_counter_ns()
                for _ in range(args.iterations):
                    result = runners[name]()
                    del result
                durations[name].append(time.perf_counter_ns() - begin)
        # Common helper's per-image field names are converted to per-graph names.
        results = {
            name: {
                key.replace("image", "graph"): value
                for key, value in summary(times, args.iterations).items()
            }
            for name, times in durations.items()
        }
        report = {
            "schema_version": 2,
            "started_at_utc": started.isoformat(),
            "workload": {
                "name": "branching_tensor",
                "shape": list(tensor.shape),
                "dtype": "float32",
                "seed": args.seed,
                "branches": BRANCHES,
                "flow_count": 1 + 3 * BRANCHES,
                "native_action_calls": 2 + 7 * BRANCHES,
                "output_count": 2 * BRANCHES,
                "pipelines": sources,
            },
            "measurement": {
                "iterations_per_sample": args.iterations,
                "samples": args.samples,
                "warmup_calls_per_backend": args.warmup,
                "clock": "perf_counter_ns",
                "includes": "full graph execution, host conversions, output materialization and release",
                "excludes": "pipeline loading, input generation, validation, warmup, thread pool creation",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "logical_cpu_count": os.cpu_count(),
                "numpy_workers": args.numpy_workers,
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
                "actions_path": os.environ.get("MORFLOW_ACTIONS_PATH"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "openblas_library": os.environ.get("MORFLOW_OPENBLAS_LIBRARY"),
                "morflow": getattr(sys.modules.get("morflow"), "__version__", None),
            },
            "validation": checks,
            "results": results,
        }
        if len(names) > 1:
            report["comparison"] = {
                f"{left.replace('-', '_')}_time_divided_by_{right.replace('-', '_')}_time": results[
                    left
                ]["mean_ms_per_graph"]
                / results[right]["mean_ms_per_graph"]
                for left, right in combinations(names, 2)
            }
        output = args.output or ROOT / "results" / (
            "branching_tensor-" + started.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
        )
        save_json(output, report)
        print(
            f"Graph: {1 + 3 * BRANCHES} flows, {BRANCHES} branches, {2 * BRANCHES} outputs"
        )
        for name, values in results.items():
            print(
                f"{name:10} {values['median_ms_per_graph']:.3f} ms/graph (median sample mean)"
            )
        print(f"JSON: {output.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    finally:
        if executor:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
