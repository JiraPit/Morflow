#!/usr/bin/env python3
"""Measure a sequential chain of matrix inversions and Cholesky with NumPy, morf-basic, or morf-blas; save validated JSON timings."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# Direct script execution needs the benchmark root to import shared helpers.
sys.path.insert(0, str(ROOT.parent))

from common import nonnegative, positive, save_json, summary  # noqa: E402

INPUT_SHAPE = (256, 256)
INVERSION_COUNT = 12
PIPELINES = {name: ROOT / f"{name}.morf" for name in ("morf-basic", "morf-blas")}


def make_input(seed):
    # A well-conditioned positive definite matrix keeps repeated inversion stable.
    values = np.random.default_rng(seed).normal(0, 0.01, INPUT_SHAPE).astype(np.float32)
    return np.ascontiguousarray(
        values @ values.T + np.eye(INPUT_SHAPE[0], dtype=np.float32)
    )


def numpy_runner(tensor):
    def run():
        value = tensor
        for _ in range(INVERSION_COUNT):
            value = np.linalg.inv(value).T
        return {"factor": np.ascontiguousarray(np.linalg.cholesky(value))}

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

    def run():
        output = pipeline.run(wrapped)
        return output if isinstance(output, dict) else {"factor": output}

    return run


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
        np.testing.assert_allclose(value, reference, atol=2e-5, rtol=2e-4, err_msg=name)
        errors[name] = float(np.max(np.abs(value - reference)))
    return {
        "passed": True,
        "max_absolute_error_by_output": errors,
        "tolerance": {"atol": 2e-5, "rtol": 2e-4},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy", action="store_true")
    parser.add_argument("--morf-basic", action="store_true")
    parser.add_argument("--morf-blas", action="store_true")
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
    try:
        started = datetime.now(timezone.utc)
        sources = {name: path.read_text() for name, path in PIPELINES.items()}
        # Generate input once. Every graph execution starts from this in-memory tensor.
        tensor = make_input(args.seed)
        reference = numpy_runner(tensor)()
        runners = {
            name: (
                numpy_runner(tensor)
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
                "name": "simple_tensor",
                "shape": list(tensor.shape),
                "dtype": "float32",
                "seed": args.seed,
                "inversion_count": INVERSION_COUNT,
                "flow_count": 1,
                "native_action_calls": 2 * INVERSION_COUNT + 1,
                "output_count": 1,
                "pipelines": sources,
            },
            "measurement": {
                "iterations_per_sample": args.iterations,
                "samples": args.samples,
                "warmup_calls_per_backend": args.warmup,
                "clock": "perf_counter_ns",
                "includes": "full graph execution, host conversions, output materialization and release",
                "excludes": "pipeline loading, input generation, validation, warmup",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "logical_cpu_count": os.cpu_count(),
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
            "simple_tensor-" + started.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
        )
        save_json(output, report)
        print(f"Graph: 1 sequential flow, {2 * INVERSION_COUNT + 1} actions, 1 output")
        for name, values in results.items():
            print(
                f"{name:10} {values['median_ms_per_graph']:.3f} ms/graph (median sample mean)"
            )
        print(f"JSON: {output.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
