#!/usr/bin/env python3
"""Compare a repeatable image-to-tensor workload using PIL/NumPy and Morflow."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
# Direct script execution needs the benchmark root to import shared helpers.
sys.path.insert(0, str(ROOT.parent))

from common import nonnegative, positive, save_json, summary, version  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pil", action="store_true", help="benchmark PIL/NumPy")
    parser.add_argument("--morflow", action="store_true", help="benchmark Morflow")
    parser.add_argument(
        "--image", type=Path, help="input file (otherwise generate seeded RGB pixels)"
    )
    parser.add_argument(
        "--width",
        type=positive,
        default=1920,
        help="generated input width (default: 1920)",
    )
    parser.add_argument(
        "--height",
        type=positive,
        default=1080,
        help="generated input height (default: 1080)",
    )
    parser.add_argument("--output-width", type=positive, default=512)
    parser.add_argument("--output-height", type=positive, default=512)
    parser.add_argument(
        "--iterations", type=positive, default=50, help="transformations per sample"
    )
    parser.add_argument(
        "--samples", type=positive, default=5, help="independent timing samples"
    )
    parser.add_argument(
        "--warmup", type=nonnegative, default=5, help="untimed calls per backend"
    )
    parser.add_argument("--seed", type=nonnegative, default=0)
    parser.add_argument(
        "--threads",
        type=positive,
        help="Morflow Rayon threads (default: library setting)",
    )
    parser.add_argument(
        "--actions-path", type=Path, help="prepared Morflow action cache"
    )
    parser.add_argument(
        "--output", type=Path, help="JSON path (default: results/<UTC timestamp>.json)"
    )
    return parser.parse_args()


def load_input(image_path, width, height, seed):
    """Decode once into owned RGB pixels; no runner retains a file handle."""
    if image_path:
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)
    return np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )


def pil_runner(
    pixels: np.ndarray, crop: tuple[int, int, int, int], width: int, height: int
):
    # Prepare the source once, just like Morflow's typed host input.
    source = Image.fromarray(pixels)
    x, y, w, h = crop
    # Match Morflow's nearest sampling: floor(output_index * float32 scale).
    # Pillow's native nearest resize uses a different pixel-center convention.

    def run():
        xs = np.minimum(
            (np.arange(width, dtype=np.float32) * np.float32(w / width)).astype(
                np.intp
            ),
            w - 1,
        )
        ys = np.minimum(
            (np.arange(height, dtype=np.float32) * np.float32(h / height)).astype(
                np.intp
            ),
            h - 1,
        )
        cropped = source.crop((x, y, x + w, y + h))
        flipped = ImageOps.mirror(cropped)
        resized = np.asarray(flipped)[ys[:, None], xs[None, :], :]
        return resized.astype(np.float32) * np.float32(1.0 / 255.0)

    return run


def morflow_runner(
    pixels: np.ndarray, crop: tuple[int, int, int, int], width: int, height: int
):
    try:
        import morflow
    except ImportError as error:
        raise RuntimeError(
            "Morflow is unavailable. Install/build the Python binding, or use --pil."
        ) from error
    try:
        # Setup only: the closure reuses this pipeline and in-memory input for every call.
        pipeline = morflow.load(str(ROOT / "pipeline.morf"))
        source = morflow.Image(pixels, color="rgb")
    except Exception as error:
        raise RuntimeError(
            "Cannot load benchmark actions. Prepare pipeline.morf with morflow prep, "
            "or package local release actions and pass --actions-path. " + str(error)
        ) from error

    def run():
        return pipeline.run(source, *crop, width, height)

    return run


def main() -> int:
    args = arguments()
    selected = [name for name in ("pil", "morflow") if getattr(args, name)] or [
        "pil",
        "morflow",
    ]
    if args.threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    if args.actions_path:
        os.environ["MORFLOW_ACTIONS_PATH"] = str(args.actions_path.resolve())
    started = datetime.now(timezone.utc)
    output = args.output or ROOT / "results" / (
        started.strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
    )
    try:
        # Phase 1: load/decode once, then construct each backend once.
        pixels = load_input(args.image, args.width, args.height, args.seed)
        h, w, _ = pixels.shape
        crop_w, crop_h = max(1, w * 3 // 4), max(1, h * 3 // 4)
        crop = ((w - crop_w) // 2, (h - crop_h) // 2, crop_w, crop_h)
        reference_run = pil_runner(pixels, crop, args.output_width, args.output_height)
        reference = reference_run()
        runners = {
            name: (
                reference_run
                if name == "pil"
                else morflow_runner(pixels, crop, args.output_width, args.output_height)
            )
            for name in selected
        }
        validation = {}
        for name, run in runners.items():
            actual = np.asarray(run())
            if actual.shape != reference.shape or actual.dtype != np.float32:
                raise RuntimeError(
                    f"{name}: unexpected output {actual.shape}, {actual.dtype}; "
                    f"expected {reference.shape}, float32."
                    + (
                        " Rebuild the current Python extension with: maturin develop --release "
                        "--manifest-path bindings/python/Cargo.toml. Python loaded "
                        + str(
                            getattr(
                                sys.modules.get("morflow._morflow"),
                                "__file__",
                                "unknown extension",
                            )
                        )
                        if name == "morflow"
                        else ""
                    )
                )
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-6)
            validation[name] = {
                "passed": True,
                "max_absolute_error": float(np.max(np.abs(actual - reference))),
            }
            for _ in range(args.warmup):
                run()
        # Phase 2: repeatedly execute the already-created runners; no disk reads or loads.
        durations = {name: [] for name in selected}
        # Alternate backend order to reduce systematic first/last measurement bias.
        for sample in range(args.samples):
            for name in selected if sample % 2 == 0 else reversed(selected):
                run = runners[name]
                begin = time.perf_counter_ns()
                for _ in range(args.iterations):
                    result = run()
                    del result
                durations[name].append(time.perf_counter_ns() - begin)
        results = {
            name: summary(values, args.iterations) for name, values in durations.items()
        }
        report = {
            "schema_version": 1,
            "started_at_utc": started.isoformat(),
            "workload": {
                "name": "crop_flip_resize_normalize",
                "input_shape": list(pixels.shape),
                "input_dtype": "uint8",
                "input_file": str(args.image.resolve()) if args.image else None,
                "seed": args.seed if not args.image else None,
                "crop_xywh": list(crop),
                "output_shape": list(reference.shape),
                "output_dtype": "float32",
                "steps": [
                    "center crop to 75%",
                    "horizontal flip",
                    "nearest resize (floor coordinates)",
                    "normalize RGB to [0, 1]",
                ],
                "pipeline": (ROOT / "pipeline.morf").read_text(),
            },
            "measurement": {
                "clock": "perf_counter_ns",
                "iterations_per_sample": args.iterations,
                "samples": args.samples,
                "warmup_calls_per_backend": args.warmup,
                "includes": "transformation calls, host call overhead, output materialization and release",
                "excludes": "input decode/generation, source wrapping, pipeline loading, validation, warmup",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "pillow": version("Pillow"),
                "morflow": getattr(sys.modules.get("morflow"), "__version__", None),
                "morflow_module": getattr(sys.modules.get("morflow"), "__file__", None),
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
                "actions_path": os.environ.get("MORFLOW_ACTIONS_PATH"),
            },
            "validation": validation,
            "results": results,
        }
        if len(selected) == 2:
            report["comparison"] = {
                "pil_time_divided_by_morflow_time": results["pil"]["mean_ms_per_image"]
                / results["morflow"]["mean_ms_per_image"]
            }
        save_json(output, report)
        for name, values in results.items():
            print(
                f"{name:8} {values['median_ms_per_image']:.3f} ms/image (median sample mean), "
                f"{values['images_per_second']:.1f} images/s"
            )
        print(f"JSON: {output.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
