"""Shared benchmark argument validation, statistics, and atomic JSON output."""

import argparse
import importlib.metadata
import json
import os
import statistics
import tempfile
from pathlib import Path


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def summary(durations: list[int], iterations: int) -> dict:
    per_call = [duration / iterations / 1_000_000 for duration in durations]
    total = sum(durations)
    return {
        "sample_total_ns": durations,
        "sample_mean_ms_per_image": per_call,
        "mean_ms_per_image": statistics.mean(per_call),
        "median_ms_per_image": statistics.median(per_call),
        "min_ms_per_image": min(per_call),
        "max_ms_per_image": max(per_call),
        "stdev_sample_mean_ms": statistics.stdev(per_call)
        if len(per_call) > 1
        else 0.0,
        "images_per_second": iterations * len(durations) * 1_000_000_000 / total,
    }


def version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_json(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
