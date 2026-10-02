"""Validate tensor graph dependencies, outputs, and backend equivalence."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from simple_tensor import benchmark as tensor


class TensorBenchmarkTests(unittest.TestCase):
    def test_tensor_pipelines_load_once_across_all_runs(self):
        data = tensor.make_input(0)
        expected = tensor.numpy_runner(data)()
        for flags, selected in [
            (["--morf-basic"], ["morf-basic"]),
            (["--morf-blas"], ["morf-blas"]),
            ([], ["morf-basic", "morf-blas"]),
        ]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                pipelines = [
                    SimpleNamespace(run=Mock(return_value=expected)) for _ in selected
                ]
                wrappers = [object() for _ in selected]
                native = SimpleNamespace(
                    load=Mock(side_effect=pipelines), Tensor=Mock(side_effect=wrappers)
                )
                output = Path(directory) / "result.json"
                argv = [
                    "tensor_benchmark.py",
                    *flags,
                    "--samples",
                    "2",
                    "--iterations",
                    "3",
                    "--warmup",
                    "2",
                    "--output",
                    str(output),
                ]
                with (
                    patch.dict(sys.modules, {"morflow": native}),
                    patch.object(sys, "argv", argv),
                    patch("builtins.print"),
                ):
                    self.assertEqual(tensor.main(), 0)
                self.assertEqual(
                    [call.args for call in native.load.call_args_list],
                    [(str(tensor.PIPELINES[name]),) for name in selected],
                )
                self.assertEqual(native.Tensor.call_count, len(selected))
                for pipeline, wrapper in zip(pipelines, wrappers, strict=True):
                    self.assertEqual(pipeline.run.call_count, 9)
                    self.assertTrue(
                        all(
                            call.args[0] is wrapper
                            for call in pipeline.run.call_args_list
                        )
                    )
                report = json.loads(output.read_text())
                self.assertEqual(
                    set(report["results"]),
                    set(selected) | ({"numpy"} if not flags else set()),
                )
                self.assertEqual(set(report["validation"]), set(report["results"]))
                if not flags:
                    self.assertEqual(len(report["comparison"]), 3)
                    self.assertIn(
                        "morf_basic_time_divided_by_morf_blas_time",
                        report["comparison"],
                    )

    def test_pipeline_variants_have_identical_sequential_graphs(self):
        basic = tensor.PIPELINES["morf-basic"].read_text()
        blas = tensor.PIPELINES["morf-blas"].read_text()
        self.assertEqual(basic.split("accept", 1)[1], blas.split("accept", 1)[1])
        self.assertIn("from linalg_basics/0.3.1 import inv, cholesky", basic)
        self.assertIn("from linalg_blas/0.1.0 import inv, cholesky", blas)
        self.assertEqual(basic.count(" >> inv"), tensor.INVERSION_COUNT)
        self.assertEqual(basic.count("emit("), 1)

    def test_numpy_chain_matches_original_matrix_factor_and_preserves_input(self):
        data = tensor.make_input(7)
        original = data.copy()
        actual = tensor.numpy_runner(data)()
        tensor.validate(actual, {"factor": np.linalg.cholesky(data)})
        np.testing.assert_array_equal(data, original)

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
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["workload"]["flow_count"], 1)
            self.assertEqual(report["workload"]["native_action_calls"], 25)
            self.assertEqual(report["workload"]["output_count"], 1)
            self.assertEqual(
                {name: path.read_text() for name, path in tensor.PIPELINES.items()},
                report["workload"]["pipelines"],
            )
            self.assertEqual(set(report["results"]), {"numpy"})
            self.assertGreater(report["results"]["numpy"]["graphs_per_second"], 0)

    @unittest.skipUnless(
        os.environ.get("MORFLOW_ACTIONS_PATH"),
        "set MORFLOW_ACTIONS_PATH for native integration",
    )
    def test_native_graph_matches_numpy_and_preserves_input(self):
        data = tensor.make_input(1)
        original = data.copy()
        expected = tensor.numpy_runner(data)()
        for backend in tensor.PIPELINES:
            with self.subTest(backend=backend):
                run = tensor.morflow_runner(data, backend)
                tensor.validate(run(), expected)
                tensor.validate(run(), expected)
        np.testing.assert_array_equal(data, original)


if __name__ == "__main__":
    unittest.main()
