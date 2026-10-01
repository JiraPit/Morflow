"""Validate tensor graph dependencies, outputs, and backend equivalence."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tensor import benchmark as tensor


class TensorBenchmarkTests(unittest.TestCase):
    def test_tensor_pipeline_load_once_across_all_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = (
                np.random.default_rng(0)
                .uniform(-2, 2, tensor.INPUT_SHAPE)
                .astype(np.float32)
            )
            expected = tensor.numpy_runner(data, tensor.BRANCHES)()
            pipeline = SimpleNamespace(run=Mock(return_value=expected))
            wrapper = object()
            native = SimpleNamespace(
                load=Mock(return_value=pipeline), Tensor=Mock(return_value=wrapper)
            )
            argv = [
                "tensor_benchmark.py",
                "--numpy",
                "--morflow",
                "--samples",
                "2",
                "--iterations",
                "3",
                "--warmup",
                "2",
                "--output",
                str(Path(directory) / "result.json"),
            ]
            with (
                patch.dict(sys.modules, {"morflow": native}),
                patch.object(sys, "argv", argv),
                patch("builtins.print"),
            ):
                self.assertEqual(tensor.main(), 0)
            native.load.assert_called_once_with(str(tensor.PIPELINE_PATH))
            native.Tensor.assert_called_once()
            self.assertEqual(pipeline.run.call_count, 9)
            self.assertTrue(
                all(call.args[0] is wrapper for call in pipeline.run.call_args_list)
            )

    def test_parallel_numpy_matches_serial_and_preserves_input(self):
        data = np.arange(35, dtype=np.float32).reshape(5, 7) / 10
        original = data.copy()
        expected = tensor.numpy_runner(data, 5)()
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = tensor.numpy_runner(data, 5, pool)()
        tensor.validate(actual, expected)
        np.testing.assert_array_equal(data, original)
        self.assertEqual(len(actual), 10)
        self.assertEqual(actual["matrix_0"].shape, (7, 5))
        self.assertEqual(actual["rows_0"].shape, (5,))

    def test_numpy_cli_graph_metadata_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(tensor.ROOT / "benchmark.py"),
                    "--numpy",
                    "--iterations",
                    "1",
                    "--samples",
                    "1",
                    "--warmup",
                    "0",
                    "--numpy-workers",
                    "2",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["workload"]["flow_count"], 49)
            self.assertEqual(report["workload"]["native_action_calls"], 114)
            self.assertEqual(report["workload"]["output_count"], 32)
            self.assertEqual(
                tensor.PIPELINE_PATH.read_text(), report["workload"]["pipeline"]
            )
            self.assertEqual(set(report["results"]), {"numpy"})
            self.assertGreater(report["results"]["numpy"]["graphs_per_second"], 0)

    @unittest.skipUnless(
        os.environ.get("MORFLOW_ACTIONS_PATH"),
        "set MORFLOW_ACTIONS_PATH for native integration",
    )
    def test_native_graph_matches_numpy_and_preserves_input(self):
        data = (
            np.random.default_rng(1)
            .uniform(-2, 2, tensor.INPUT_SHAPE)
            .astype(np.float32)
        )
        original = data.copy()
        expected = tensor.numpy_runner(data, tensor.BRANCHES)()
        run = tensor.morflow_runner(data)
        tensor.validate(run(), expected)
        tensor.validate(run(), expected)
        np.testing.assert_array_equal(data, original)


if __name__ == "__main__":
    unittest.main()
