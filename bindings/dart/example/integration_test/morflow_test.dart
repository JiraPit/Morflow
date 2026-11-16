// Runs on a device or emulator: every action comes from the pack libraries
// bundled into the app, since mobile apps have no action cache.
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:morflow/morflow.dart' as morflow;
import 'package:morflow/morflow.dart' show MorflowException, MorflowTensor, TensorInput;

Matcher throwsMorflow(String message) =>
    throwsA(isA<MorflowException>().having((e) => e.message, 'message', contains(message)));

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const identity = 'accept Tensor \$t\n\$t >> base/latest/identity >> emit';

  test('bundled base pack runs synchronously and asynchronously', () async {
    final pipeline = morflow.fromStr(identity);
    final input = Float32List.fromList([1.0, 2.5, -3.0]);
    expect(pipeline.runSync<MorflowTensor>(input).toFloat32Array(), [1.0, 2.5, -3.0]);
    expect((await pipeline.run<MorflowTensor>(input)).toFloat32Array(), [1.0, 2.5, -3.0]);
  });

  test('exact versions select the bundled pack', () {
    final pipeline = morflow.fromStr('from base/0.3.3 import identity\naccept Tensor \$t\n\$t >> identity >> emit');
    expect(pipeline.runSync<MorflowTensor>(Float32List.fromList([4])).toFloat32Array(), [4]);
    expect(
      () => morflow.fromStr('from base/0.0.1 import identity\naccept Tensor \$t\n\$t >> identity >> emit'),
      throwsMorflow('is not bundled; this app bundles base 0.3.3'),
    );
  });

  test('a whole-pack import runs an audio pipeline', () async {
    final pipeline = morflow.fromStr(r'''
      import audio_basics/latest
      accept Audio $audio
      $audio >> gain(linear=2.0) >> to_wav >> emit
    ''');
    final wav = await pipeline.run<MorflowTensor>(
      TensorInput(data: Float32List(2 * 100), shape: [2, 100], payloadType: 'audio', sampleRate: 44100),
    );
    final bytes = wav.toBuffer();
    expect(String.fromCharCodes(bytes.sublist(0, 4)), 'RIFF');
    expect(String.fromCharCodes(bytes.sublist(8, 12)), 'WAVE');
  });

  test('Composites round-trip', () {
    final pipeline = morflow.fromStr('accept Composite \$pair\n\$pair >> emit');
    final [first as MorflowTensor, second as MorflowTensor] = pipeline.runSync<List<Object>>([
      Float64List.fromList([1.5]),
      Int64List.fromList([7]),
    ]);
    expect(first.toFloat64Array(), [1.5]);
    expect(second.toInt64Array(), [7]);
  });

  test('packs that are not bundled are reported', () {
    expect(
      () => morflow.fromStr('import tensor_basics/latest\naccept Tensor \$t\n\$t >> transpose >> emit'),
      throwsMorflow('No prepared catalog for tensor_basics/latest'),
    );
  });

  test('many concurrent runs', () async {
    final pipeline = morflow.fromStr(identity);
    final outputs = await Future.wait([
      for (var i = 0; i < 50; i++) pipeline.run<MorflowTensor>(Float32List.fromList([i.toDouble()])),
    ]);
    for (var i = 0; i < 50; i++) {
      expect(outputs[i].toFloat32Array(), [i.toDouble()]);
    }
  });
}
