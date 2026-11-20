// Runs every demo's pipeline through the real engine: flutter test
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:morflow/morflow.dart' as morflow;
import 'package:morflow_showcase/demos.dart';

morflow.Pipeline pipeline(String name) => morflow.fromStr(File('pipelines/$name.morf').readAsStringSync());

void main() {
  test('every pipeline compiles', () {
    for (final name in pipelineNames) {
      pipeline(name).close();
    }
  });

  test('the image effects keep the picture size', () async {
    final rgb = Uint8List.fromList(List.generate(imageSize * imageSize * 3, (i) => (i * 37) % 256));
    for (final name in ['blur', 'edges', 'vivid']) {
      final run = await runImage(pipeline(name), rgb, 2);
      expect(run.output.shape, [imageSize, imageSize, 3], reason: name);
      expect(run.output.dtype, 'u8', reason: name);
      expect(run.timings.runs, hasLength(2));
      expect(toRgba(run.output.data, 3), hasLength(imageSize * imageSize * 4));
    }
  });

  test('the audio clean-up keeps the length and peaks at -1 dB', () async {
    final run = await runAudio(pipeline('clean_audio'), makeRecording(), 1);
    final samples = run.output.toFloat32Array();
    expect(samples, hasLength(sampleRate * recordingSeconds));
    final peak = samples.map((s) => s.abs()).reduce((a, b) => a > b ? a : b);
    expect(peak, closeTo(0.891, 0.01));
  });

  test('matrix multiplication matches Dart', () async {
    final run = await runMatmul(pipeline('matmul'), 128, 1);
    expect(run.checked, isTrue);
    expect(run.output.shape, [128, 128]);
    final a = randomMatrix(64, 1), b = randomMatrix(64, 2);
    final dart = dartMatmul((a, b, 64));
    final engine = (await pipeline('matmul').run<morflow.MorflowTensor>([
      morflow.TensorInput(data: a, shape: const [64, 64]),
      morflow.TensorInput(data: b, shape: const [64, 64]),
    ])).toFloat32Array();
    for (var i = 0; i < dart.length; i++) {
      expect(engine[i], closeTo(dart[i], 1e-3));
    }
  });

  test('timings', () {
    final t = Timings([5, 1, 3]);
    expect((t.first, t.best, t.median), (5, 1, 3));
    expect(realTimeFactor(50), 100);
  });
}
