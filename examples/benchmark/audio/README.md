# Audio processing benchmark

Compare **NumPy/SciPy** and **morf-basic** on the same in-memory 16-bit PCM WAV. Each execution performs:

1. Decode the WAV to planar float32 samples.
2. Resample to 16 kHz.
3. High-pass filter at 80 Hz to remove rumble.
4. Peak-normalize to 0.9.
5. From the cleaned audio, compute a magnitude spectrogram (`n_fft=512`, hop 160) and re-encode a 16-bit WAV.

[`morf-basic.morf`](morf-basic.morf) runs these steps with `audio_basics/0.2.3` in three flows: one prepares the cleaned audio, and two consume it. There are 6 native action calls and two named outputs, `spectrogram` and `wav`.

The NumPy/SciPy backend implements the same algorithms as the Morflow actions, so their outputs can be checked against each other: the windowed-sinc resampler, the RBJ cookbook high-pass in transposed direct form II (`scipy.signal.lfilter`), and a Hann-windowed STFT without padding. It does not use SciPy's own resampling or STFT functions, which produce different results.

## Setup

From the repository root:

```sh
python3 -m pip install -r examples/benchmark/audio/requirements.txt
```

Install the current Morflow Python binding and prepare the actions. See [binding and local action setup](../image/README.md#setup). When the actions are published, prepare the pipeline with:

```sh
morflow prep examples/benchmark/audio/morf-basic.morf
```

## Run

```sh
# Both backends; no selection flags means both.
python3 examples/benchmark/audio/benchmark.py --actions-path target/release/actions

# Select one backend.
python3 examples/benchmark/audio/benchmark.py --numpy
python3 examples/benchmark/audio/benchmark.py --morf-basic --actions-path target/release/actions

# Use a recording, more samples, and four Morflow workers.
python3 examples/benchmark/audio/benchmark.py --input speech.wav \
  --iterations 20 --samples 7 --warmup 5 --threads 4 \
  --actions-path target/release/actions --output results/audio-benchmark.json
```

Without `--input`, the benchmark generates a 10-second, 2-channel, 44.1 kHz clip of tones, low-frequency rumble and noise. Change it with `--seconds`, `--channels`, `--sample-rate` and `--seed`. An `--input` file must be uncompressed 16-bit PCM. Defaults are three warmup calls, five samples and ten clips per sample. Backend order alternates between samples.

`--threads` sets Morflow's Rayon worker count before loading. NumPy and SciPy run on the calling thread.

## Load once, run repeatedly

The WAV is generated or read exactly once into memory. The Morflow backend calls `morflow.load` once and then passes the same WAV bytes to every `run()`.

```text
Read or generate the WAV once → load the pipeline once → run → run → run → …
```

Validation, warmup and timed runs reuse these instances. Every iteration decodes the original bytes. Iterations do not read files or reload the pipeline.

## Validation and timing

Before timing, each backend's outputs are checked against the NumPy/SciPy reference. The spectrogram must have the same shape and float32 dtype, with magnitudes within an absolute tolerance of `1e-2`. The WAV must have the same header and length, with every sample within three 16-bit steps. Both implementations run the high-pass filter in single precision, and its poles sit close to the unit circle, so small rounding differences can grow to two steps. The report records the largest difference and the number of differing WAV samples. Failed validation stops the run without writing a report.

Timing uses `perf_counter_ns` and includes WAV decoding, processing, the spectrogram, WAV encoding, Python/native overhead, output materialization and release. Input generation or reading, pipeline loading, validation and warmup are excluded.

## JSON reports

Reports use schema version 2 and are saved under `audio/results/` unless `--output` specifies a path. They include the input description, the processing settings, the pipeline source, timing settings, package versions, platform and thread configuration, validation details and raw sample durations. Results give the mean, median, minimum, maximum and standard deviation of the sample means in milliseconds per clip, clips per second, and seconds of audio processed per second. With both backends selected, `numpy_time_divided_by_morf_basic_time` above 1 means `morf-basic` was faster in that run.

## Tests

```sh
MORFLOW_ACTIONS_PATH="$PWD/target/release/actions" \
  python3 -m unittest discover -s examples/benchmark/audio -t examples/benchmark
```

Tests verify one pipeline load across repeated runs, the pipeline arguments against the reference, the reference's filtering, normalization and output shapes, JSON metadata, and 16-bit input checking. With `MORFLOW_ACTIONS_PATH` set, they also check that the native pipeline matches the reference and leaves its input unchanged.
