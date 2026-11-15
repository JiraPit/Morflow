"""Validate the audio pipeline, its NumPy/SciPy reference, and backend equivalence."""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from audio import benchmark as audio


def band_energy(samples, sample_rate, low, high):
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    return spectrum[(frequencies >= low) & (frequencies <= high)].sum()


class AudioBenchmarkTests(unittest.TestCase):
    def test_audio_pipeline_loads_once_across_all_runs(self):
        data = audio.make_input(0.5, 2, 44100, 0)
        expected = audio.numpy_runner(data)()
        for flags, selected in [(["--morf-basic"], ["morf-basic"]), ([], ["morf-basic"])]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                pipeline = SimpleNamespace(run=Mock(return_value=expected))
                native = SimpleNamespace(load=Mock(return_value=pipeline))
                output = Path(directory) / "result.json"
                argv = [
                    "audio_benchmark.py",
                    *flags,
                    "--seconds",
                    "0.5",
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
                    self.assertEqual(audio.main(), 0)
                self.assertEqual(
                    [call.args for call in native.load.call_args_list],
                    [(str(audio.PIPELINES[name]),) for name in selected],
                )
                # One validation run, two warmup runs, and 2 samples of 3 runs.
                self.assertEqual(pipeline.run.call_count, 9)
                inputs = [call.args[0] for call in pipeline.run.call_args_list]
                self.assertTrue(all(value == data for value in inputs))
                report = json.loads(output.read_text())
                self.assertEqual(
                    set(report["results"]),
                    set(selected) | ({"numpy"} if not flags else set()),
                )
                self.assertEqual(set(report["validation"]), set(report["results"]))
                if not flags:
                    self.assertEqual(
                        list(report["comparison"]), ["numpy_time_divided_by_morf_basic_time"]
                    )

    def test_pipeline_arguments_match_the_reference(self):
        source = audio.PIPELINES["morf-basic"].read_text()
        self.assertIn(
            "from audio_basics/0.2.3 import to_audio, resample, biquad_filter, normalize, stft, to_wav",
            source,
        )
        self.assertIn(f"resample(to_rate={audio.TARGET_RATE})", source)
        self.assertIn(f'biquad_filter(type="highpass", freq={audio.HIGHPASS_HZ})', source)
        self.assertIn(f'normalize(mode="peak", target_peak={audio.TARGET_PEAK})', source)
        self.assertIn(f"stft(n_fft={audio.N_FFT}, hop_size={audio.HOP_SIZE})", source)
        self.assertEqual(re.findall(r'emit\("(\w+)"\)', source), ["spectrogram", "wav"])

    def test_reference_removes_rumble_normalizes_and_preserves_input(self):
        data = audio.make_input(2.0, 2, 44100, 3)
        original = bytes(data)
        outputs = audio.numpy_runner(data)()
        self.assertEqual(data, original)

        source, _ = audio.decode(data)
        result, rate = audio.decode(outputs["wav"])
        self.assertEqual(rate, audio.TARGET_RATE)
        self.assertEqual(result.shape, (2, 32000))
        self.assertAlmostEqual(float(np.abs(result).max()), audio.TARGET_PEAK, delta=1e-4)
        for channel in range(2):
            # The rumble below the cutoff loses most of its energy; the tone keeps it.
            rumble = (20, 40)
            tone = (200, 360)
            before = band_energy(source[channel], 44100, *rumble) / band_energy(
                source[channel], 44100, *tone
            )
            after = band_energy(result[channel], rate, *rumble) / band_energy(
                result[channel], rate, *tone
            )
            self.assertLess(after, before / 10)

        spectrum = outputs["spectrogram"]
        frames = (32000 - audio.N_FFT) // audio.HOP_SIZE + 1
        self.assertEqual(spectrum.shape, (2, audio.N_FFT // 2 + 1, frames))
        self.assertEqual(spectrum.dtype, np.float32)

    def test_numpy_cli_metadata_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(audio.ROOT / "benchmark.py"),
                    "--numpy",
                    "--seconds",
                    "1",
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
            self.assertEqual(report["workload"]["flow_count"], 3)
            self.assertEqual(report["workload"]["native_action_calls"], 6)
            self.assertEqual(report["workload"]["output_count"], 2)
            self.assertEqual(report["workload"]["input"]["channels"], 2)
            self.assertEqual(report["workload"]["input"]["sample_rate"], 44100)
            self.assertEqual(
                {name: path.read_text() for name, path in audio.PIPELINES.items()},
                report["workload"]["pipelines"],
            )
            self.assertEqual(set(report["results"]), {"numpy"})
            numpy = report["results"]["numpy"]
            self.assertGreater(numpy["clips_per_second"], 0)
            self.assertAlmostEqual(
                numpy["audio_seconds_per_second"], numpy["clips_per_second"], places=6
            )

    def test_input_file_must_be_16_bit_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eight_bit.wav"
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(1)
                file.setframerate(8000)
                file.writeframes(bytes(800))
            path.write_bytes(buffer.getvalue())
            argv = ["audio_benchmark.py", "--numpy", "--input", str(path)]
            with patch.object(sys, "argv", argv), patch("builtins.print") as printed:
                self.assertEqual(audio.main(), 1)
            self.assertIn("16-bit PCM", str(printed.call_args))

    @unittest.skipUnless(
        os.environ.get("MORFLOW_ACTIONS_PATH"),
        "set MORFLOW_ACTIONS_PATH for native integration",
    )
    def test_native_pipeline_matches_reference_and_preserves_input(self):
        for seconds, channels in [(2.0, 2), (1.5, 1)]:
            data = audio.make_input(seconds, channels, 44100, 1)
            original = bytes(data)
            expected = audio.numpy_runner(data)()
            with self.subTest(seconds=seconds, channels=channels):
                run = audio.morflow_runner(data, "morf-basic")
                audio.validate(run(), expected)
                audio.validate(run(), expected)
            self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
