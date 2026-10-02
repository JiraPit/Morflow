"""Small correctness and CLI checks; these do not assert performance."""

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
from PIL import Image

from image import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_image_and_pipelines_load_once_across_all_runs(self):
        for flags, selected in [
            (["--morf-basic"], ["morf-basic"]),
            (["--morf-opencv"], ["morf-opencv"]),
            ([], ["morf-basic", "morf-opencv"]),
        ]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "input.png"
                pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
                Image.fromarray(pixels).save(image)
                expected = benchmark.pil_runner(pixels, (1, 1, 6, 6), 4, 4)()
                pipelines = [
                    SimpleNamespace(run=Mock(return_value=expected)) for _ in selected
                ]
                wrappers = [object() for _ in selected]
                native = SimpleNamespace(
                    load=Mock(side_effect=pipelines), Image=Mock(side_effect=wrappers)
                )
                output = Path(directory) / "result.json"
                argv = [
                    "benchmark.py",
                    *flags,
                    "--image",
                    str(image),
                    "--output-width",
                    "4",
                    "--output-height",
                    "4",
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
                    patch.object(benchmark.Image, "open", wraps=Image.open) as opened,
                    patch("builtins.print"),
                ):
                    self.assertEqual(benchmark.main(), 0)
                opened.assert_called_once_with(image)
                self.assertEqual(
                    [call.args for call in native.load.call_args_list],
                    [(str(benchmark.PIPELINES[name]),) for name in selected],
                )
                self.assertEqual(native.Image.call_count, len(selected))
                for pipeline, wrapper in zip(pipelines, wrappers, strict=True):
                    self.assertEqual(pipeline.run.call_count, 9)
                    self.assertTrue(
                        all(
                            call.args == (wrapper, 1, 1, 6, 6, 4, 4)
                            for call in pipeline.run.call_args_list
                        )
                    )
                report = json.loads(output.read_text())
                self.assertEqual(
                    set(report["results"]),
                    set(selected) | ({"pil"} if not flags else set()),
                )
                self.assertEqual(report["schema_version"], 2)
                if not flags:
                    self.assertEqual(len(report["comparison"]), 3)
                    self.assertIn(
                        "morf_basic_time_divided_by_morf_opencv_time",
                        report["comparison"],
                    )

    def test_pipeline_variants_share_the_same_workload(self):
        basic = benchmark.PIPELINES["morf-basic"].read_text()
        opencv = benchmark.PIPELINES["morf-opencv"].read_text()
        self.assertEqual(basic.split("accept", 1)[1], opencv.split("accept", 1)[1])
        self.assertIn("from image_opencv/0.1.0 import resize", opencv)
        self.assertIn("from image_basics/0.3.1 import crop, flip", opencv)

    def test_reference_coordinate_convention(self):
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        actual = benchmark.pil_runner(pixels, (1, 1, 2, 2), 3, 3)()
        expected = pixels[1:3, 1:3][:, ::-1][
            np.array([0, 0, 1])[:, None], np.array([0, 0, 1])[None, :]
        ]
        np.testing.assert_array_equal(
            actual, expected.astype(np.float32) * np.float32(1 / 255)
        )

    def test_nearest_coordinate_rounding_boundary(self):
        basic = benchmark.nearest_indices(2, 82, "basics")
        opencv = benchmark.nearest_indices(2, 82, "opencv")
        self.assertEqual(basic[41], 0)
        self.assertEqual(opencv[41], 1)
        self.assertEqual(np.count_nonzero(basic != opencv), 1)

    def test_pil_cli_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(benchmark.ROOT / "benchmark.py"),
                    "--pil",
                    "--width",
                    "17",
                    "--height",
                    "11",
                    "--output-width",
                    "7",
                    "--output-height",
                    "5",
                    "--iterations",
                    "2",
                    "--samples",
                    "2",
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
            self.assertEqual(set(report["results"]), {"pil"})
            self.assertNotIn("comparison", report)
            self.assertEqual(report["workload"]["output_shape"], [5, 7, 3])
            self.assertEqual(len(report["results"]["pil"]["sample_total_ns"]), 2)
            self.assertTrue(report["validation"]["pil"]["passed"])

    def test_invalid_iterations_do_not_write_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(benchmark.ROOT / "benchmark.py"),
                    "--pil",
                    "--iterations",
                    "0",
                    "--output",
                    str(output),
                ],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        os.environ.get("MORFLOW_ACTIONS_PATH"),
        "set MORFLOW_ACTIONS_PATH for native integration",
    )
    def test_native_matches_reference(self):
        for width, height, out_w, out_h in [
            (1, 1, 3, 2),
            (17, 11, 7, 5),
            (64, 48, 16, 12),
            (9, 7, 23, 19),
            (8, 8, 85, 31),
            (3, 4, 82, 2),
        ]:
            with self.subTest(input=(width, height), output=(out_w, out_h)):
                pixels = np.random.default_rng(3).integers(
                    0, 256, (height, width, 3), dtype=np.uint8
                )
                w, h = max(1, width * 3 // 4), max(1, height * 3 // 4)
                crop = ((width - w) // 2, (height - h) // 2, w, h)
                backends = ["morf-basic"]
                if os.environ.get("MORFLOW_PLUGINS_PATH"):
                    backends.append("morf-opencv")
                for backend in backends:
                    with self.subTest(backend=backend):
                        run = benchmark.morflow_runner(
                            pixels, crop, out_w, out_h, backend
                        )
                        native_reference = benchmark.pil_runner(
                            pixels,
                            crop,
                            out_w,
                            out_h,
                            sampling="opencv" if backend == "morf-opencv" else "basics",
                        )()
                        np.testing.assert_allclose(
                            run(), native_reference, rtol=0, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            run(), native_reference, rtol=0, atol=1e-6
                        )
                np.testing.assert_array_equal(
                    pixels,
                    np.random.default_rng(3).integers(
                        0, 256, pixels.shape, dtype=np.uint8
                    ),
                )


class NativeProcessLifetimeTests(unittest.TestCase):
    def test_large_opencv_benchmark_exits_after_native_workers_start(self):
        import subprocess

        if not os.environ.get("MORFLOW_PLUGINS_PATH") or not os.environ.get(
            "MORFLOW_ACTIONS_PATH"
        ):
            self.skipTest(
                "Requires prepared actions, OpenCV plugin and runtime libraries"
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(benchmark.ROOT / "benchmark.py"),
                    "--morf-opencv",
                    "--width",
                    "1920",
                    "--height",
                    "1080",
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
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["validation"]["morf-opencv"]["passed"])


if __name__ == "__main__":
    unittest.main()
