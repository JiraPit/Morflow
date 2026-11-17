// Runs on a device or emulator: every action comes from the pack libraries
// bundled into the app, since mobile apps have no action cache.
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:morflow/morflow.dart' as morflow;
import 'package:morflow/morflow.dart'
    show MorflowException, MorflowTensor, TensorInput;

Matcher throwsMorflow(String message) => throwsA(
  isA<MorflowException>().having(
    (e) => e.message,
    'message',
    contains(message),
  ),
);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const identity = 'accept Tensor \$t\n\$t >> base/latest/identity >> emit';

  test('bundled base pack runs synchronously and asynchronously', () async {
    final pipeline = morflow.fromStr(identity);
    final input = Float32List.fromList([1.0, 2.5, -3.0]);
    expect(pipeline.runSync<MorflowTensor>(input).toFloat32Array(), [
      1.0,
      2.5,
      -3.0,
    ]);
    expect((await pipeline.run<MorflowTensor>(input)).toFloat32Array(), [
      1.0,
      2.5,
      -3.0,
    ]);
  });

  test('exact versions select the bundled pack', () {
    final pipeline = morflow.fromStr(
      'from base/0.3.4 import identity\naccept Tensor \$t\n\$t >> identity >> emit',
    );
    expect(
      pipeline
          .runSync<MorflowTensor>(Float32List.fromList([4]))
          .toFloat32Array(),
      [4],
    );
    expect(
      () => morflow.fromStr(
        'from base/0.0.1 import identity\naccept Tensor \$t\n\$t >> identity >> emit',
      ),
      throwsMorflow('is not bundled; this app bundles base 0.3.4'),
    );
  });

  test('a whole-pack import runs an audio pipeline', () async {
    final pipeline = morflow.fromStr(r'''
      import audio_basics/latest
      accept Audio $audio
      $audio >> gain(linear=2.0) >> to_wav >> emit
    ''');
    final wav = await pipeline.run<MorflowTensor>(
      TensorInput(
        data: Float32List(2 * 100),
        shape: [2, 100],
        payloadType: 'audio',
        sampleRate: 44100,
      ),
    );
    final bytes = wav.toBuffer();
    expect(String.fromCharCodes(bytes.sublist(0, 4)), 'RIFF');
    expect(String.fromCharCodes(bytes.sublist(8, 12)), 'WAVE');
  });

  test('image_basics converts image payloads', () {
    final pipeline = morflow.fromStr(r"""
      from base/latest import to_tensor
      from image_basics/latest import to_image
      accept Image $image
      $image >> to_tensor >> to_image >> emit
    """);
    final output = pipeline.runSync<MorflowTensor>(
      TensorInput(
        data: Float32List(64),
        shape: [8, 8, 1],
        payloadType: 'image',
        colorSpace: 'grayscale',
      ),
    );
    expect(output.shape, [8, 8]);
  });

  test('linalg_basics returns QR as a Composite', () {
    final pipeline = morflow.fromStr(r"""
      import linalg_basics/latest
      accept Tensor[3,2] $matrix
      $matrix >> qr >> $parts
      $parts[0] >> emit("q")
      $parts[1] >> emit("r")
      $parts >> matmul >> emit("reconstructed")
    """);
    final data = Float32List.fromList([1, 0, 0, 1, 1, 1]);
    final out = pipeline.runSyncAll<MorflowTensor>(
      TensorInput(data: data, shape: [3, 2]),
    );
    expect(out['q']!.shape, [3, 2]);
    expect(out['r']!.shape, [2, 2]);
    final reconstructed = out['reconstructed']!.toFloat32Array();
    for (var i = 0; i < data.length; i++) {
      expect((reconstructed[i] - data[i]).abs(), lessThan(1e-5));
    }
  });

  test('math_basics adds a Composite pair', () async {
    final pipeline = morflow.fromStr(r"""
      import math_basics/latest
      accept Composite $pair
      $pair >> add >> emit
    """);
    final output = await pipeline.run<MorflowTensor>([
      TensorInput(data: Int32List.fromList([1, 2, 3, 4, 5, 6]), shape: [2, 3]),
      Int32List.fromList([10, 20, 30]),
    ]);
    expect(output.toInt32Array(), [11, 22, 33, 14, 25, 36]);
  });

  test('nn_basics takes a scalar argument', () {
    final pipeline = morflow.fromStr(r"""
      import nn_basics/latest
      accept Scalar $value
      $value >> relu >> emit
    """);
    expect(pipeline.runSyncArgs<MorflowTensor>([-3.0]).toFloat32Array(), [0.0]);
  });

  test('tensor_basics reshapes', () {
    final pipeline = morflow.fromStr(r"""
      import tensor_basics/latest
      accept Tensor $data
      $data >> reshape(shape="2, 1000") >> emit
    """);
    final output = pipeline.runSync<MorflowTensor>(
      TensorInput(data: Float32List(2000), shape: [4, 500]),
    );
    expect(output.shape, [2, 1000]);
  });

  test('tensor_stats sums integers to i64', () {
    final pipeline = morflow.fromStr(r"""
      import tensor_stats/latest
      accept Tensor $data
      $data >> sum(axis=0) >> emit
    """);
    final output = pipeline.runSync<MorflowTensor>(
      TensorInput(data: Int32List.fromList([1, 2, 3, 4]), shape: [2, 2]),
    );
    expect(output.dtype, 'i64');
    expect(output.toInt64Array(), [4, 6]);
  });

  test('plugins are rejected on mobile', () {
    expect(
      () => morflow.fromStr(
        'plugin opencv-bridge/0.1.3\nfrom image_opencv/0.1.3 import resize\n'
        'accept Tensor[6,8,3] \$image\n\$image >> resize(4,3) >> emit',
      ),
      throwsMorflow('Plugins are not available on Android and iOS'),
    );
  });

  test('Composites round-trip', () {
    final pipeline = morflow.fromStr('accept Composite \$pair\n\$pair >> emit');
    final [first as MorflowTensor, second as MorflowTensor] = pipeline
        .runSync<List<Object>>([
          Float64List.fromList([1.5]),
          Int64List.fromList([7]),
        ]);
    expect(first.toFloat64Array(), [1.5]);
    expect(second.toInt64Array(), [7]);
  });

  test('packs that are not bundled are reported', () {
    expect(
      () => morflow.fromStr(
        'import linalg_blas/latest\naccept Tensor \$t\n\$t >> matmul >> emit',
      ),
      throwsMorflow('No prepared catalog for linalg_blas/latest'),
    );
  });

  test('many concurrent runs', () async {
    final pipeline = morflow.fromStr(identity);
    final outputs = await Future.wait([
      for (var i = 0; i < 50; i++)
        pipeline.run<MorflowTensor>(Float32List.fromList([i.toDouble()])),
    ]);
    for (var i = 0; i < 50; i++) {
      expect(outputs[i].toFloat32Array(), [i.toDouble()]);
    }
  });
}
