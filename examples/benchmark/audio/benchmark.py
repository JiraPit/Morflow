#!/usr/bin/env python3
"""Measure WAV decoding, resampling, filtering, a spectrogram, and WAV encoding with NumPy/SciPy or morf-basic; save validated JSON timings."""

from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import time
import wave
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent
# Direct script execution needs the benchmark root to import shared helpers.
sys.path.insert(0, str(ROOT.parent))

from common import nonnegative, positive, save_json, summary, version  # noqa: E402

PIPELINES = {"morf-basic": ROOT / "morf-basic.morf"}
# These match the arguments in morf-basic.morf.
TARGET_RATE = 16000
HIGHPASS_HZ = 80.0
HIGHPASS_Q = 0.707
TARGET_PEAK = 0.9
N_FFT = 512
HOP_SIZE = 160
RESAMPLE_HALF_LENGTH = 8
# Both implementations run the high-pass filter in single precision, and its
# poles sit close to the unit circle, so small rounding differences grow to at
# most two 16-bit steps in the re-encoded WAV (measured over clips of 1 to 60
# seconds and 1 to 6 channels). Spectrogram magnitudes differ by under 0.004.
SPECTROGRAM_ATOL = 1e-2
WAV_LSB_TOLERANCE = 3

f32 = np.float32


def make_input(seconds, channels, sample_rate, seed):
    """A 16-bit PCM WAV with tones, low-frequency rumble, and noise per channel."""
    rng = np.random.default_rng(seed)
    count = int(round(seconds * sample_rate))
    t = np.arange(count) / sample_rate
    signal = np.stack(
        [
            0.3 * np.sin(2 * np.pi * (220 + 110 * channel) * t)
            + 0.2 * np.sin(2 * np.pi * (30 - 5 * channel) * t)
            + 0.05 * rng.standard_normal(count)
            for channel in range(channels)
        ],
        axis=1,
    )
    pcm = np.round(np.clip(signal, -1, 1) * 32767).astype("<i2")
    return encode_pcm(pcm, sample_rate)


def encode_pcm(pcm, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(pcm.shape[1])
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def describe(data):
    with wave.open(io.BytesIO(data)) as file:
        if file.getsampwidth() != 2 or file.getcomptype() != "NONE":
            raise ValueError("The input WAV must be uncompressed 16-bit PCM")
        return {
            "channels": file.getnchannels(),
            "sample_rate": file.getframerate(),
            "frames": file.getnframes(),
            "seconds": file.getnframes() / file.getframerate(),
        }


def round_half_away(values):
    # Rust's f32::round rounds halves away from zero; NumPy rounds them to even.
    return np.sign(values) * np.floor(np.abs(values) + f32(0.5))


def decode(data):
    with wave.open(io.BytesIO(data)) as file:
        channels, rate, frames = file.getnchannels(), file.getframerate(), file.getnframes()
        pcm = np.frombuffer(file.readframes(frames), "<i2").reshape(frames, channels).T
    return pcm.astype(f32) / f32(32768.0), rate


def resample(samples, source_rate, target_rate):
    """Windowed-sinc interpolation, as in audio_basics resample."""
    ratio = f32(target_rate) / f32(source_rate)
    count = samples.shape[-1]
    if count == 0 or abs(f32(source_rate) - f32(target_rate)) < 0.1:
        return samples
    length = int(round_half_away(f32(count) * ratio))
    cutoff = ratio * f32(0.95) if ratio < 1 else f32(0.95)
    half = RESAMPLE_HALF_LENGTH
    centers = np.arange(length, dtype=f32) / ratio
    taps = np.floor(centers).astype(np.int64)[:, None] - half + np.arange(2 * half + 2)
    last = np.ceil(centers).astype(np.int64)[:, None] + half
    offset = taps.astype(f32) - centers[:, None]
    position = offset / f32(half)
    used = (taps >= 0) & (taps < count) & (taps <= last) & (np.abs(position) <= 1)
    window = f32(0.5) * (f32(1) + np.cos(f32(np.pi) * position))
    kernel = np.sinc(cutoff * offset).astype(f32) * cutoff
    weights = np.where(used, window * kernel, f32(0))
    totals = weights.sum(axis=1)
    values = (samples[:, np.clip(taps, 0, count - 1)] * weights).sum(axis=-1)
    return np.where(np.abs(totals) > 1e-6, values / totals, f32(0)).astype(f32)


def highpass(samples, sample_rate, frequency, q):
    """RBJ cookbook high-pass in transposed direct form II, as in audio_basics biquad_filter."""
    f0 = np.clip(f32(frequency), f32(10), f32(sample_rate) * f32(0.499))
    w0 = f32(2) * f32(np.pi) * f0 / f32(sample_rate)
    cos_w0, alpha = np.cos(w0), np.sin(w0) / (f32(2) * max(f32(q), f32(0.01)))
    a0 = f32(1) + alpha
    b = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2], dtype=f32) / a0
    a = np.array([1, -2 * cos_w0 / a0, (1 - alpha) / a0], dtype=f32)
    return lfilter(b, a, samples, axis=-1).astype(f32)


def normalize_peak(samples, target):
    peak = f32(np.abs(samples).max()) if samples.size else f32(0)
    if peak < 1e-8:
        return samples
    return (samples * (f32(target) / peak)).astype(f32)


def spectrogram(samples, n_fft, hop):
    """Hann-windowed magnitude STFT as [channels, bins, frames], as in audio_basics stft."""
    index = np.arange(n_fft, dtype=f32)
    window = f32(0.5) * (f32(1) - np.cos(f32(2) * f32(np.pi) * index / f32(n_fft - 1)))
    frames = (samples.shape[-1] - n_fft) // hop + 1
    starts = np.arange(frames)[:, None] * hop + np.arange(n_fft)
    fft_size = 1 << (max(n_fft, 16) - 1).bit_length()
    magnitude = np.abs(np.fft.rfft(samples[:, starts] * window, n=fft_size, axis=-1))
    return np.ascontiguousarray(magnitude.astype(f32).transpose(0, 2, 1))


def encode(samples, sample_rate):
    quantized = round_half_away(np.clip(samples, -1, 1) * f32(32767))
    return encode_pcm(np.clip(quantized, -32768, 32767).astype("<i2").T, sample_rate)


def numpy_runner(data):
    def run():
        samples, rate = decode(data)
        samples = resample(samples, rate, TARGET_RATE)
        samples = highpass(samples, TARGET_RATE, HIGHPASS_HZ, HIGHPASS_Q)
        samples = normalize_peak(samples, TARGET_PEAK)
        return {
            "spectrogram": spectrogram(samples, N_FFT, HOP_SIZE),
            "wav": encode(samples, TARGET_RATE),
        }

    return run


def morflow_runner(data, backend):
    try:
        import morflow

        # Setup once; all validation, warmup, and measured calls reuse this instance.
        pipeline = morflow.load(str(PIPELINES[backend]))
    except (ImportError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Install the current Morflow Python binding and prepare the pipeline actions. "
            + str(error)
        ) from error

    def run():
        return pipeline.run(data)

    return run


def validate(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise RuntimeError(
            "Output names do not match the pipeline. Rebuild the Python binding if it is stale."
        )
    spectrum = np.asarray(actual["spectrogram"])
    reference = expected["spectrogram"]
    if spectrum.shape != reference.shape or spectrum.dtype != np.float32:
        raise RuntimeError(
            f"spectrogram: expected {reference.shape}, float32; got {spectrum.shape}, {spectrum.dtype}"
        )
    np.testing.assert_allclose(
        spectrum, reference, atol=SPECTROGRAM_ATOL, rtol=0, err_msg="spectrogram"
    )
    encoded, reference_wav = bytes(actual["wav"]), expected["wav"]
    if len(encoded) != len(reference_wav) or encoded[:44] != reference_wav[:44]:
        raise RuntimeError("wav: the header or length differs from the reference")
    difference = np.abs(
        np.frombuffer(encoded[44:], "<i2").astype(np.int32)
        - np.frombuffer(reference_wav[44:], "<i2").astype(np.int32)
    )
    largest = int(difference.max()) if difference.size else 0
    if largest > WAV_LSB_TOLERANCE:
        raise RuntimeError(f"wav: samples differ from the reference by up to {largest} steps")
    return {
        "passed": True,
        "max_absolute_error_by_output": {
            "spectrogram": float(np.max(np.abs(spectrum - reference))),
            "wav_steps": largest,
        },
        "wav_samples_differing": int(np.count_nonzero(difference)),
        "tolerance": {"spectrogram_atol": SPECTROGRAM_ATOL, "wav_steps": WAV_LSB_TOLERANCE},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy", action="store_true")
    parser.add_argument("--morf-basic", action="store_true")
    parser.add_argument("--input", type=Path, help="16-bit PCM WAV file to use instead of generated audio")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--channels", type=positive, default=2)
    parser.add_argument("--sample-rate", type=positive, default=44100)
    parser.add_argument("--threads", type=positive, help="Morflow Rayon worker count")
    parser.add_argument("--iterations", type=positive, default=10)
    parser.add_argument("--samples", type=positive, default=5)
    parser.add_argument("--warmup", type=nonnegative, default=3)
    parser.add_argument("--seed", type=nonnegative, default=0)
    parser.add_argument("--actions-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    available = ("numpy", "morf-basic")
    names = [
        name for name in available if getattr(args, name.replace("-", "_"))
    ] or list(available)
    if args.threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    if args.actions_path:
        os.environ["MORFLOW_ACTIONS_PATH"] = str(args.actions_path.resolve())
    try:
        if args.seconds <= 0:
            raise ValueError("--seconds must be greater than zero")
        started = datetime.now(timezone.utc)
        sources = {name: path.read_text() for name, path in PIPELINES.items()}
        # Read or generate the WAV once. Every run decodes these same bytes.
        data = (
            args.input.read_bytes()
            if args.input
            else make_input(args.seconds, args.channels, args.sample_rate, args.seed)
        )
        clip = describe(data)
        reference = numpy_runner(data)()
        runners = {
            name: numpy_runner(data) if name == "numpy" else morflow_runner(data, name)
            for name in names
        }
        checks = {}
        for name, run in runners.items():
            checks[name] = validate(run(), reference)
            for _ in range(args.warmup):
                run()
        del reference
        # Repeated execution only: runners never load pipelines or re-read inputs.
        durations = {name: [] for name in names}
        for sample in range(args.samples):
            for name in names if sample % 2 == 0 else reversed(names):
                begin = time.perf_counter_ns()
                for _ in range(args.iterations):
                    result = runners[name]()
                    del result
                durations[name].append(time.perf_counter_ns() - begin)
        # Common helper's per-image field names are converted to per-clip names.
        results = {}
        for name, times in durations.items():
            values = {
                key.replace("image", "clip"): value
                for key, value in summary(times, args.iterations).items()
            }
            values["audio_seconds_per_second"] = values["clips_per_second"] * clip["seconds"]
            results[name] = values
        report = {
            "schema_version": 2,
            "started_at_utc": started.isoformat(),
            "workload": {
                "name": "audio",
                "source": str(args.input) if args.input else "generated",
                "seed": None if args.input else args.seed,
                "input": clip,
                "target_sample_rate": TARGET_RATE,
                "highpass_hz": HIGHPASS_HZ,
                "target_peak": TARGET_PEAK,
                "n_fft": N_FFT,
                "hop_size": HOP_SIZE,
                "flow_count": 3,
                "native_action_calls": 6,
                "output_count": 2,
                "pipelines": sources,
            },
            "measurement": {
                "iterations_per_sample": args.iterations,
                "samples": args.samples,
                "warmup_calls_per_backend": args.warmup,
                "clock": "perf_counter_ns",
                "includes": "WAV decoding, processing, spectrogram, WAV encoding, output materialization and release",
                "excludes": "pipeline loading, input generation or reading, validation, warmup",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": version("scipy"),
                "logical_cpu_count": os.cpu_count(),
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
                "actions_path": os.environ.get("MORFLOW_ACTIONS_PATH"),
                "morflow": getattr(sys.modules.get("morflow"), "__version__", None),
            },
            "validation": checks,
            "results": results,
        }
        if len(names) > 1:
            report["comparison"] = {
                f"{left.replace('-', '_')}_time_divided_by_{right.replace('-', '_')}_time": results[
                    left
                ]["mean_ms_per_clip"]
                / results[right]["mean_ms_per_clip"]
                for left, right in combinations(names, 2)
            }
        output = args.output or ROOT / "results" / (
            "audio-" + started.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
        )
        save_json(output, report)
        print(
            f"Clip: {clip['channels']} channels, {clip['sample_rate']} Hz, {clip['seconds']:.2f} s"
        )
        for name, values in results.items():
            print(
                f"{name:10} {values['median_ms_per_clip']:.3f} ms/clip (median sample mean), "
                f"{values['audio_seconds_per_second']:.1f} s of audio per second"
            )
        print(f"JSON: {output.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, AssertionError, wave.Error) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
