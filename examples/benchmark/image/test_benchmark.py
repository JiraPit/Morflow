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
    def test_image_and_pipeline_load_once_across_all_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
            Image.fromarray(pixels).save(image)
            expected = benchmark.pil_runner(pixels, (1, 1, 6, 6), 4, 4)()
            pipeline = SimpleNamespace(run=Mock(return_value=expected))
            wrapper = object()
            native = SimpleNamespace(
                load=Mock(return_value=pipeline), Image=Mock(return_value=wrapper)
            )
            argv = [
                "benchmark.py",
                "--pil",
                "--morflow",
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
                str(Path(directory) / "result.json"),
            ]
            with (
                patch.dict(sys.modules, {"morflow": native}),
                patch.object(sys, "argv", argv),
                patch.object(benchmark.Image, "open", wraps=Image.open) as opened,
                patch("builtins.print"),
            ):
                self.assertEqual(benchmark.main(), 0)
            opened.assert_called_once_with(image)
            native.load.assert_called_once()
            native.Image.assert_called_once()
            self.assertEqual(
                pipeline.run.call_count, 9
            )  # validation + 2 warmups + 2 × 3 measurements
            self.assertTrue(
                all(call.args[0] is wrapper for call in pipeline.run.call_args_list)
            )

    def test_reference_coordinate_convention(self):
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        actual = benchmark.pil_runner(pixels, (1, 1, 2, 2), 3, 3)()
        expected = pixels[1:3, 1:3][:, ::-1][
            np.array([0, 0, 1])[:, None], np.array([0, 0, 1])[None, :]
        ]
        np.testing.assert_array_equal(
            actual, expected.astype(np.float32) * np.float32(1 / 255)
        )

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
        ]:
            with self.subTest(input=(width, height), output=(out_w, out_h)):
                pixels = np.random.default_rng(3).integers(
                    0, 256, (height, width, 3), dtype=np.uint8
                )
                w, h = max(1, width * 3 // 4), max(1, height * 3 // 4)
                crop = ((width - w) // 2, (height - h) // 2, w, h)
                expected = benchmark.pil_runner(pixels, crop, out_w, out_h)()
                actual = benchmark.morflow_runner(pixels, crop, out_w, out_h)()
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(
                    pixels,
                    np.random.default_rng(3).integers(
                        0, 256, pixels.shape, dtype=np.uint8
                    ),
                )


if __name__ == "__main__":
    unittest.main()
