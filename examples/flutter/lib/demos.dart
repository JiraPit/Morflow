// What each demo runs and measures, apart from the UI, so it can be tested.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:morflow/morflow.dart' as morflow;

/// The pipelines in pipelines/, by file name.
const pipelineNames = ['blur', 'edges', 'vivid', 'clean_audio', 'matmul'];

const imageSize = 720;
const sampleRate = 44100;
const recordingSeconds = 5;
const benchmarkRuns = 20;

/// The timings of a benchmark, in milliseconds.
class Timings {
  Timings(this.runs) : assert(runs.isNotEmpty);
  final List<double> runs;

  double get first => runs.first;
  double get best => runs.reduce(math.min);
  double get median {
    final sorted = [...runs]..sort();
    return sorted[sorted.length ~/ 2];
  }
}

/// A demo's last output and how long its runs took.
class Run {
  Run(this.output, this.timings, {this.checked});
  final morflow.MorflowTensor output;
  final Timings timings;

  /// Whether the output matched a check made in Dart, where there is one.
  final bool? checked;
}

/// Runs [pipeline] on [input] [count] times, one after another.
Future<Run> timedRuns(morflow.Pipeline pipeline, Object input, int count) async {
  final runs = <double>[];
  late morflow.MorflowTensor output;
  for (var i = 0; i < count; i++) {
    final watch = Stopwatch()..start();
    output = await pipeline.run<morflow.MorflowTensor>(input);
    runs.add(watch.elapsedMicroseconds / 1000);
  }
  return Run(output, Timings(runs));
}

// Images -------------------------------------------------------------------

/// RGB bytes from RGBA bytes.
Uint8List rgbaToRgb(Uint8List rgba) {
  final rgb = Uint8List(rgba.length ~/ 4 * 3);
  for (var p = 0, q = 0; p < rgba.length; p += 4, q += 3) {
    rgb[q] = rgba[p];
    rgb[q + 1] = rgba[p + 1];
    rgb[q + 2] = rgba[p + 2];
  }
  return rgb;
}

/// RGBA bytes from pixels of 1 (gray), 3 (RGB) or 4 (RGBA) channels.
Uint8List toRgba(Uint8List pixels, int channels) {
  final rgba = Uint8List(pixels.length ~/ channels * 4);
  for (var p = 0, q = 0; q < rgba.length; p += channels, q += 4) {
    rgba[q] = pixels[p];
    rgba[q + 1] = pixels[p + (channels > 1 ? 1 : 0)];
    rgba[q + 2] = pixels[p + (channels > 2 ? 2 : 0)];
    rgba[q + 3] = channels == 4 ? pixels[p + 3] : 255;
  }
  return rgba;
}

/// An RGB picture through an image pipeline.
Future<Run> runImage(morflow.Pipeline pipeline, Uint8List rgb, int count) =>
    timedRuns(pipeline, morflow.TensorInput(data: rgb, shape: const [imageSize, imageSize, 3], dtype: 'u8'), count);

double megapixelsPerSecond(double ms) => imageSize * imageSize / 1e6 / (ms / 1000);

// Audio --------------------------------------------------------------------

/// A mono recording with loud and quiet phrases and some hiss.
Float32List makeRecording() {
  final random = math.Random(3);
  final samples = Float32List(sampleRate * recordingSeconds);
  for (var i = 0; i < samples.length; i++) {
    final t = i / sampleRate;
    // Phrases that swell and fade, every other one much louder.
    final phrase = math.pow(math.sin(math.pi * (t % 1.25) / 1.25), 2) * (0.15 + 0.6 * ((t ~/ 1.25) % 2));
    final voice =
        math.sin(2 * math.pi * 220 * t) +
        0.5 * math.sin(2 * math.pi * 440 * t) +
        0.25 * math.sin(2 * math.pi * 660 * t);
    final hiss = (random.nextDouble() * 2 - 1) * 0.06;
    samples[i] = phrase * voice * 0.5 + hiss;
  }
  return samples;
}

Future<Run> runAudio(morflow.Pipeline pipeline, Float32List samples, int count) => timedRuns(
  pipeline,
  morflow.TensorInput(data: samples, shape: [1, samples.length], payloadType: 'audio', sampleRate: sampleRate),
  count,
);

/// How many times faster than playback the pipeline ran.
double realTimeFactor(double ms) => recordingSeconds * 1000 / ms;

// Math ---------------------------------------------------------------------

Float32List randomMatrix(int n, int seed) {
  final random = math.Random(seed);
  return Float32List.fromList(List.generate(n * n, (_) => random.nextDouble() * 2 - 1));
}

/// The same product in plain Dart, for comparison (run it in an isolate).
Float32List dartMatmul((Float32List, Float32List, int) args) {
  final (a, b, n) = args;
  final c = Float32List(n * n);
  for (var i = 0; i < n; i++) {
    for (var k = 0; k < n; k++) {
      final aik = a[i * n + k];
      for (var j = 0; j < n; j++) {
        c[i * n + j] += aik * b[k * n + j];
      }
    }
  }
  return c;
}

/// Two random n×n matrices multiplied, with a few entries checked in Dart.
Future<Run> runMatmul(morflow.Pipeline pipeline, int n, int count) async {
  final a = randomMatrix(n, 1), b = randomMatrix(n, 2);
  final run = await timedRuns(pipeline, [
    morflow.TensorInput(data: a, shape: [n, n]),
    morflow.TensorInput(data: b, shape: [n, n]),
  ], count);
  final c = run.output.toFloat32Array();
  var checked = c.length == n * n;
  for (final (i, j) in [(0, 0), (n - 1, n - 1), (n ~/ 2, n ~/ 3)]) {
    var expected = 0.0;
    for (var k = 0; k < n; k++) {
      expected += a[i * n + k] * b[k * n + j];
    }
    checked &= (c[i * n + j] - expected).abs() < 1e-3 * math.max(1, expected.abs());
  }
  return Run(run.output, run.timings, checked: checked);
}

double gflops(int n, double ms) => 2.0 * n * n * n / (ms / 1000) / 1e9;
