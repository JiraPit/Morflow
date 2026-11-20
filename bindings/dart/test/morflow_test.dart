// Port of bindings/js/tests/morflow.test.mjs, followed by Dart-specific tests.
// Tests that execute actions need MORFLOW_ACTIONS_PATH pointing at a prepared
// action cache (see scripts/prepare_test_actions.py); the rest run anywhere.
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:morflow/morflow.dart' as morflow;
import 'package:morflow/morflow.dart' show MorflowException, MorflowTensor, TensorInput;
import 'package:morflow/src/version.dart';
import 'package:test/test.dart';

/// Actions come from a prepared action cache or from bundled pack libraries.
final Object? needsActions =
    Platform.environment['MORFLOW_ACTIONS_PATH'] == null && Platform.environment['MORFLOW_PACKS_PATH'] == null
    ? 'set MORFLOW_ACTIONS_PATH to a prepared action cache, or MORFLOW_PACKS_PATH to pack libraries'
    : null;
final Object? needsPlugins = Platform.environment['MORFLOW_PLUGINS_PATH'] == null
    ? 'set MORFLOW_PLUGINS_PATH to run plugin tests'
    : null;

Matcher throwsMorflow(Pattern message) =>
    throwsA(isA<MorflowException>().having((e) => e.message, 'message', contains(message)));

Uint8List f32Bytes(List<double> values) => Float32List.fromList(values).buffer.asUint8List();

String ascii(Uint8List bytes, int start, int end) => String.fromCharCodes(bytes.sublist(start, end));

void main() {
  group('Node.js parity', () {
    test('fromStr compilation and metadata', () {
      final pipeline = morflow.fromStr(r'''
        accept Audio $audio_in
        accept IntArg $rate = 44100

        $audio_in >> base/latest/identity >> emit
      ''');
      expect(pipeline.params.length, 2);
      expect(pipeline.params[0], 'audio_in');
      expect(pipeline.params[1], 'rate');
    }, skip: needsActions);

    test('synchronous execution with Float32List', () {
      final pipeline = morflow.fromStr(r'''
        accept Tensor $tensor
        $tensor >> base/latest/identity >> emit
      ''');
      final output = pipeline.runSync<MorflowTensor>(Float32List.fromList([1.0, 2.5, -3.0, 4.25]));
      expect(output.dtype, 'f32');
      expect(output.shape, [4]);
      expect(output.length, 4);
      expect(output.toFloat32Array(), [1.0, 2.5, -3.0, 4.25]);
    }, skip: needsActions);

    test('asynchronous execution with Future', () async {
      final pipeline = morflow.fromStr(r'''
        accept Tensor $tensor
        $tensor >> base/latest/identity >> emit
      ''');
      final output = await pipeline.run<MorflowTensor>(Float32List.fromList([10.0, 20.0, 30.0]));
      expect(output.dtype, 'f32');
      expect(output.shape, [3]);
      expect(output.toFloat32Array(), [10.0, 20.0, 30.0]);
    }, skip: needsActions);

    test('multi-dimensional TensorInput', () {
      final pipeline = morflow.fromStr(r'''
        accept Tensor $img
        $img[1:3, :] >> base/latest/identity >> emit
      ''');
      final matrix = Float32List.fromList([for (var i = 0; i < 16; i++) i.toDouble()]);
      final output = pipeline.runSync<MorflowTensor>(TensorInput(data: matrix.buffer, shape: [4, 4], dtype: 'f32'));
      expect(output.shape, [2, 4]);
      expect(output.length, 8);
      final values = output.toFloat32Array();
      expect(values[0], 4.0); // row 1, col 0
      expect(values[7], 11.0); // row 2, col 3
    }, skip: needsActions);

    test('multiple named outputs (runSyncToMap & runToMap)', () async {
      final pipeline = morflow.fromStr(r'''
        accept Tensor $audio
        $audio[0:2] >> base/latest/identity >> emit("low")
        $audio[2:4] >> base/latest/identity >> emit("high")
      ''');
      final input = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final syncOut = pipeline.runSyncToMap<MorflowTensor>(input);
      expect(syncOut['low']!.toFloat32Array(), [1.0, 2.0]);
      expect(syncOut['high']!.toFloat32Array(), [3.0, 4.0]);
      final asyncOut = await pipeline.runToMap<MorflowTensor>(input);
      expect(asyncOut['low']!.toFloat32Array(), [1.0, 2.0]);
      expect(asyncOut['high']!.toFloat32Array(), [3.0, 4.0]);
    }, skip: needsActions);

    test('error handling on invalid pipeline syntax', () {
      expect(() => morflow.fromStr(r'accept Tensor $x >>> broken'), throwsMorflow('Parse error'));
    });

    test('audio to_audio and to_wav pipeline', () async {
      final pipeline = morflow.fromStr(r'''
        import audio_basics/latest
        accept Bytes $data
        $data >> to_audio(channels=2, sample_rate=44100, dtype="i16") >> gain(linear=2.0) >> to_wav >> emit
      ''');
      final pcm = ByteData(8)
        ..setInt16(0, 16384, Endian.little)
        ..setInt16(2, -8192, Endian.little)
        ..setInt16(4, 16384, Endian.little)
        ..setInt16(6, -8192, Endian.little);
      final output = await pipeline.run<MorflowTensor>(pcm.buffer.asUint8List());
      final wav = output.toBuffer();
      expect(wav.length, 44 + 8);
      expect(ascii(wav, 0, 4), 'RIFF');
      expect(ascii(wav, 8, 12), 'WAVE');
    }, skip: needsActions);

    final reshape = r'''
      import tensor_basics/latest
      accept Tensor $data
      $data >> reshape(shape="2, 1000") >> emit
    ''';
    final counting = Float32List.fromList([for (var i = 0; i < 2000; i++) i.toDouble()]);

    test('bare rank-2 array is a plain tensor', () {
      final output = morflow
          .fromStr(reshape)
          .runSync<MorflowTensor>(TensorInput(data: counting.buffer, shape: [4, 500], dtype: 'f32'));
      expect(output.shape, [2, 1000]);
    }, skip: needsActions);

    test('payloadType tensor forces a plain tensor', () {
      final output = morflow
          .fromStr(reshape)
          .runSync<MorflowTensor>(
            TensorInput(data: counting.buffer, shape: [4, 500], dtype: 'f32', payloadType: 'tensor'),
          );
      expect(output.shape, [2, 1000]);
      expect(output.toFloat32Array()[0], 0);
      expect(output.toFloat32Array()[1999], 1999);
    }, skip: needsActions);

    test('payloadType audio forces an audio payload', () async {
      final pipeline = morflow.fromStr(r'''
        import audio_basics/latest
        accept Audio $audio
        $audio >> to_wav >> emit
      ''');
      final output = await pipeline.run<MorflowTensor>(
        TensorInput(
          data: Uint8List(2 * 1000 * 4),
          shape: [2, 1000],
          dtype: 'f32',
          payloadType: 'audio',
          sampleRate: 48000,
        ),
      );
      final wav = output.toBuffer();
      expect(ascii(wav, 0, 4), 'RIFF');
      expect(ascii(wav, 8, 12), 'WAVE');
    }, skip: needsActions);

    test('payloadType image forces an image payload', () {
      final pipeline = morflow.fromStr(r'''
        from base/latest import to_tensor
        from image_basics/latest import to_image
        accept Image $image
        $image >> to_tensor >> to_image >> emit
      ''');
      final output = pipeline.runSync<MorflowTensor>(
        TensorInput(
          data: Uint8List(8 * 8 * 4),
          shape: [8, 8, 1],
          dtype: 'f32',
          payloadType: 'image',
          colorSpace: 'grayscale',
        ),
      );
      expect(output.shape, [8, 8]);
    }, skip: needsActions);

    final identity = r'''
      accept Tensor $data
      $data >> base/latest/identity >> emit
    ''';

    test('payloadType rejects unknown values', () {
      final pipeline = morflow.fromStr(identity);
      expect(
        () => pipeline.runSync(
          TensorInput(data: Uint8List(8 * 8 * 4), shape: [8, 8], dtype: 'f32', payloadType: 'bogus'),
        ),
        throwsMorflow('Unknown payloadType'),
      );
    }, skip: needsActions);

    test('colorSpace rejects unknown values', () {
      final pipeline = morflow.fromStr(identity);
      expect(
        () => pipeline.runSync(
          TensorInput(
            data: Uint8List(8 * 8 * 3 * 4),
            shape: [8, 8, 3],
            dtype: 'f32',
            payloadType: 'image',
            colorSpace: 'cmyk',
          ),
        ),
        throwsMorflow('Unknown color space'),
      );
    }, skip: needsActions);

    test('bare rank-2 array is not inferred as audio', () {
      final pipeline = morflow.fromStr(r'''
        from audio_basics/latest import to_wav
        accept Audio $audio
        $audio >> to_wav >> emit
      ''');
      final data = Uint8List(2 * 1000 * 4);
      expect(
        () => pipeline.runSync(TensorInput(data: data, shape: [2, 1000], dtype: 'f32')),
        throwsMorflow('expected Audio'),
      );
      final output = pipeline.runSync<MorflowTensor>(
        TensorInput(data: data, shape: [2, 1000], dtype: 'f32', payloadType: 'audio', sampleRate: 44100),
      );
      expect(ascii(output.toBuffer(), 0, 4), 'RIFF');
    }, skip: needsActions);

    test('bare rank-3 array runs as a plain tensor', () {
      final pipeline = morflow.fromStr(r'''
        from image_basics/latest import to_image
        accept Tensor $image
        $image >> to_image >> emit
      ''');
      final output = pipeline.runSync<MorflowTensor>(
        TensorInput(data: Uint8List(8 * 8 * 3), shape: [8, 8, 3], dtype: 'u8'),
      );
      expect(output.shape, [8, 8, 3]);
      expect(output.dtype, 'u8');
    }, skip: needsActions);

    test('scalar param accepts a plain number', () {
      final pipeline = morflow.fromStr(r'''
        import nn_basics/latest
        accept Scalar $value
        $value >> relu >> emit
      ''');
      expect(pipeline.runSyncArgs<MorflowTensor>([-3.0]).toFloat32Array(), [0.0]);
    }, skip: needsActions);

    test('IntArg positional parameter with default', () async {
      final pipeline = morflow.fromStr(r'''
        import audio_basics/latest
        accept Bytes $data
        accept IntArg $rate = 48000
        $data >> to_audio(channels=2, sample_rate=$rate, dtype="i16") >> to_wav >> emit
      ''');
      final pcm = Uint8List(8);
      expect(ascii((await pipeline.runArgs<MorflowTensor>([pcm])).toBuffer(), 0, 4), 'RIFF');
      expect(ascii((await pipeline.runArgs<MorflowTensor>([pcm, 22050])).toBuffer(), 0, 4), 'RIFF');
    }, skip: needsActions);

    test('StrArg, BoolArg and Scalar positional parameters', () {
      final pipeline = morflow.fromStr(r'''
        import base/latest
        from audio_basics/latest import to_wav
        accept Audio $audio
        accept FloatArg $linear = 1.0
        accept BoolArg $routed = true
        accept Scalar $mix = 0.0
        $audio >> to_wav >> emit
      ''');
      final outputs = pipeline.runSyncToMapArgs<MorflowTensor>([
        TensorInput(data: Uint8List(2 * 64 * 4), shape: [2, 64], dtype: 'f32', payloadType: 'audio', sampleRate: 44100),
        2.0,
        false,
        0.5,
      ]);
      expect(outputs, isNotEmpty);
    }, skip: needsActions);

    test('exact version and required imports', () {
      final pipeline = morflow.fromStr(
        'from base/0.3.4 import identity\n'
        'accept Tensor \$data\n'
        '\$data >> identity >> emit',
      );
      expect(pipeline.runSync<MorflowTensor>(Float32List.fromList([1, 2])).toFloat32Array(), [1, 2]);
      expect(
        () => morflow.fromStr('accept Tensor \$data\n\$data >> identity >> emit'),
        throwsMorflow('not declared by the imports'),
      );
    }, skip: needsActions);

    test('select QR Composite components', () {
      final pipeline = morflow.fromStr(r'''
        import linalg_basics/latest
        accept Tensor[3,2] $matrix
        $matrix >> qr >> $parts
        $parts[0] >> emit("q")
        $parts[1] >> emit("r")
            $parts >> matmul >> emit("reconstructed")
      ''');
      final data = Float32List.fromList([1, 0, 0, 1, 1, 1]);
      final out = pipeline.runSyncToMap<MorflowTensor>(TensorInput(data: data.buffer, shape: [3, 2], dtype: 'f32'));
      expect(out['q']!.shape, [3, 2]);
      expect(out['r']!.shape, [2, 2]);
      expect(out['reconstructed']!.shape, [3, 2]);
      final reconstructed = out['reconstructed']!.toFloat32Array();
      for (var i = 0; i < data.length; i++) {
        expect((reconstructed[i] - data[i]).abs(), lessThan(1e-5));
      }
      final q = out['q']!.toFloat32Array();
      final r = out['r']!.toFloat32Array();
      for (var i = 0; i < 6; i++) {
        final row = i ~/ 2, col = i % 2;
        expect((q[row * 2] * r[col] + q[row * 2 + 1] * r[2 + col] - data[i]).abs(), lessThan(1e-5));
      }
    }, skip: needsActions);

    test('every dtype round-trips through a typed list', () {
      final pipeline = morflow.fromStr(identity);
      final inputs = <String, TypedData>{
        'f32': Float32List.fromList([1.5, -2]),
        'f64': Float64List.fromList([1.5, -2]),
        'i8': Int8List.fromList([-3, 4]),
        'i16': Int16List.fromList([-300, 400]),
        'i32': Int32List.fromList([-70000, 80000]),
        'i64': Int64List.fromList([-(1 << 40), 1 << 40]),
        'u32': Uint32List.fromList([1, 4000000000]),
        'u64': Uint64List.fromList([1, 1 << 62]),
      };
      for (final MapEntry(key: dtype, value: input) in inputs.entries) {
        final output = pipeline.runSync<MorflowTensor>(input);
        expect(output.dtype, dtype);
        expect(output.shape, [2]);
        expect(output.toTypedArray(), input);
      }
      final bytes = pipeline.runSync<MorflowTensor>(
        TensorInput(data: Uint8List.fromList([7, 8]), shape: [2], dtype: 'u8'),
      );
      expect(bytes.dtype, 'u8');
      expect(bytes.toTypedArray(), [7, 8]);
    }, skip: needsActions);

    test('integer sum returns i64', () {
      final pipeline = morflow.fromStr(r'''
        import tensor_stats/latest
        accept Tensor $data
        $data >> sum(axis=0) >> emit
      ''');
      final output = pipeline.runSync<MorflowTensor>(
        TensorInput(data: Int32List.fromList([1, 2, 3, 4]), shape: [2, 2]),
      );
      expect(output.dtype, 'i64');
      expect(output.toInt64Array(), [4, 6]);
    }, skip: needsActions);

    test('unknown dtype is rejected', () {
      final pipeline = morflow.fromStr(identity);
      expect(
        () => pipeline.runSync(TensorInput(data: Uint8List(8), shape: [2], dtype: 'f16')),
        throwsMorflow('Unknown dtype'),
      );
    }, skip: needsActions);

    test('a List is a Composite for two-tensor arithmetic', () async {
      final pipeline = morflow.fromStr(r'''
        import math_basics/latest
        accept Composite $pair
        $pair >> add >> emit
      ''');
      final a = TensorInput(data: Int32List.fromList([1, 2, 3, 4, 5, 6]), shape: [2, 3]);
      final b = Int32List.fromList([10, 20, 30]);
      final output = await pipeline.run<MorflowTensor>([a, b]);
      expect(output.dtype, 'i32');
      expect(output.shape, [2, 3]);
      expect(output.toInt32Array(), [11, 22, 33, 14, 25, 36]);
    }, skip: needsActions);

    test('a Composite output is a List', () {
      final pipeline = morflow.fromStr(r'''
        accept Composite $pair
        $pair >> emit
      ''');
      final [first as MorflowTensor, second as MorflowTensor] = pipeline.runSync<List<Object>>([
        Float64List.fromList([1.5, 2.5]),
        Int64List.fromList([7]),
      ]);
      expect(first.dtype, 'f64');
      expect(first.toFloat64Array(), [1.5, 2.5]);
      expect(second.dtype, 'i64');
      expect(second.toInt64Array(), [7]);
    });

    test('a raw byte buffer is not a Composite component', () {
      final pipeline = morflow.fromStr(r'''
        accept Composite $pair
        $pair >> emit
      ''');
      expect(
        () => pipeline.runSync([
          Uint8List.fromList([1, 2]),
        ]),
        throwsArgumentError,
      );
    });

    test('declared plugin works through file and string APIs', () async {
      final directory = Directory.systemTemp.createTempSync('morflow-plugin-');
      const source =
          'plugin opencv-bridge/0.1.3\n'
          'from image_opencv/0.1.3 import resize\n'
          'accept Tensor[6,8,3] \$image\n'
          '\$image >> resize(4,3,filter="nearest") >> emit';
      final file = File('${directory.path}/plugin.morf')..writeAsStringSync(source);
      try {
        final values = Float32List(6 * 8 * 3)..fillRange(0, 6 * 8 * 3, 0.5);
        final input = TensorInput(data: values.buffer, shape: [6, 8, 3], dtype: 'f32');
        for (final pipeline in [morflow.load(file.path), morflow.fromStr(source)]) {
          final outputs = [
            pipeline.runSync<MorflowTensor>(input),
            ...await Future.wait([pipeline.run<MorflowTensor>(input), pipeline.run<MorflowTensor>(input)]),
          ];
          for (final output in outputs) {
            expect(output.shape, [3, 4, 3]);
            expect(output.toFloat32Array().every((value) => value == 0.5), isTrue);
          }
        }
      } finally {
        directory.deleteSync(recursive: true);
      }
    }, skip: needsActions ?? needsPlugins);
  });

  // Pipelines without actions run anywhere, so these cover the binding itself.
  const echo = 'accept Tensor \$data\n\$data >> emit';
  const echoBytes = 'accept Bytes \$data\n\$data >> emit';
  const echoComposite = 'accept Composite \$pair\n\$pair >> emit';
  const twoOutputs = 'accept Tensor \$t\n\$t[0:1] >> emit("a")\n\$t[1:2] >> emit("b")';

  group('values', () {
    test('every dtype round-trips', () {
      final pipeline = morflow.fromStr(echo);
      final inputs = <String, TypedData>{
        'f32': Float32List.fromList([1.5, -2]),
        'f64': Float64List.fromList([1.5, -2]),
        'i8': Int8List.fromList([-3, 4]),
        'i16': Int16List.fromList([-300, 400]),
        'i32': Int32List.fromList([-70000, 80000]),
        'i64': Int64List.fromList([-(1 << 40), 1 << 40]),
        'u32': Uint32List.fromList([1, 4000000000]),
        'u64': Uint64List.fromList([1, 1 << 62]),
      };
      for (final MapEntry(key: dtype, value: input) in inputs.entries) {
        final output = pipeline.runSync<MorflowTensor>(input);
        expect(output.dtype, dtype, reason: dtype);
        expect(output.toTypedArray(), input, reason: dtype);
      }
      final u8 = pipeline.runSync<MorflowTensor>(
        TensorInput(data: Uint8List.fromList([7, 8]), shape: [2], dtype: 'u8'),
      );
      expect(u8.toTypedArray(), [7, 8]);
    });

    test('TensorInput infers dtype from typed data and defaults to f32', () {
      final pipeline = morflow.fromStr(echo);
      expect(pipeline.runSync<MorflowTensor>(TensorInput(data: Int16List.fromList([1, -1]), shape: [2])).dtype, 'i16');
      expect(pipeline.runSync<MorflowTensor>(TensorInput(data: f32Bytes([1, 2]), shape: [2])).dtype, 'f32');
    });

    test('views into a larger buffer send only their own bytes', () {
      final backing = Float32List.fromList([9, 1, 2, 9]);
      final view = Float32List.sublistView(backing, 1, 3);
      expect(morflow.fromStr(echo).runSync<MorflowTensor>(view).toFloat32Array(), [1, 2]);
    });

    test('tensor metadata and views', () {
      final output = morflow
          .fromStr(echo)
          .runSync<MorflowTensor>(TensorInput(data: Float32List.fromList(List.filled(6, 1)), shape: [2, 3]));
      expect(output.rank, 2);
      expect(output.length, 6);
      expect(output.shape, [2, 3]);
      expect(() => output.shape.add(1), throwsUnsupportedError);
      expect(output.toBuffer().lengthInBytes, 24);
      expect(output.toString(), 'MorflowTensor(dtype: f32, shape: [2, 3])');
    });

    test('raw bytes and no input', () {
      final pipeline = morflow.fromStr(echoBytes);
      final bytes = pipeline.runSync<MorflowTensor>(Uint8List.fromList([1, 2, 3]));
      expect(bytes.dtype, 'u8');
      expect(bytes.toBuffer(), [1, 2, 3]);
      final empty = pipeline.runSync<MorflowTensor>();
      expect(empty.shape, [0]);
      expect(empty.toBuffer(), isEmpty);
    });

    test('nested Composites round-trip', () {
      final pipeline = morflow.fromStr(echoComposite);
      final output = pipeline.runSync<List<Object>>([
        [
          Float32List.fromList([1]),
        ],
        Float32List.fromList([2]),
      ]);
      expect(output[0], isA<List<Object>>());
      expect(((output[0] as List)[0] as MorflowTensor).toFloat32Array(), [1]);
      expect((output[1] as MorflowTensor).toFloat32Array(), [2]);
    });

    test('numbers, strings and booleans bind as the Node.js binding does', () {
      final pipeline = morflow.fromStr(echoBytes);
      String bound(Object value) => utf8.decode(pipeline.runSyncArgs<MorflowTensor>([value]).toBuffer());
      expect(bound(3), '3');
      expect(bound(3.0), '3');
      expect(bound(-3), '-3');
      expect(bound(2.5), '2.5');
      expect(bound(44100), '44100');
      expect(bound('hann'), 'hann');
      expect(bound('ünïcödé'), 'ünïcödé');
      expect(bound(true), 'true');
      expect(bound(false), 'false');
    });

    test('named outputs', () async {
      final pipeline = morflow.fromStr(twoOutputs);
      final input = Float32List.fromList([1, 2]);
      for (final outputs in [
        pipeline.runSyncToMap<MorflowTensor>(input),
        await pipeline.runToMap<MorflowTensor>(input),
        pipeline.runSyncToMapArgs<MorflowTensor>([input]),
        await pipeline.runToMapArgs<MorflowTensor>([input]),
        // The old names still work.
        // ignore: deprecated_member_use_from_same_package
        pipeline.runSyncAll<MorflowTensor>(input),
        // ignore: deprecated_member_use_from_same_package
        await pipeline.runAll<MorflowTensor>(input),
        // ignore: deprecated_member_use_from_same_package
        pipeline.runSyncAllArgs<MorflowTensor>([input]),
        // ignore: deprecated_member_use_from_same_package
        await pipeline.runAllArgs<MorflowTensor>([input]),
      ]) {
        expect(outputs.keys, unorderedEquals(['a', 'b']));
        expect(outputs['a']!.toFloat32Array(), [1]);
        expect(outputs['b']!.toFloat32Array(), [2]);
      }
    });

    test('single-output runs reject several outputs', () async {
      final pipeline = morflow.fromStr(twoOutputs);
      final input = Float32List.fromList([1, 2]);
      expect(() => pipeline.runSync(input), throwsMorflow('Expected a single emitted output'));
      await expectLater(pipeline.run(input), throwsMorflow('Expected a single emitted output'));
    });
  });

  group('input validation', () {
    test('engine rejections match the Node.js binding', () {
      final pipeline = morflow.fromStr(echo);
      expect(
        () => pipeline.runSync(TensorInput(data: Uint8List(8), shape: [2], dtype: 'f16')),
        throwsMorflow("Unknown dtype 'f16'"),
      );
      expect(
        () => pipeline.runSync(TensorInput(data: Uint8List(6), shape: [2], dtype: 'f32')),
        throwsMorflow('Data buffer size is not a multiple of 4 for a f32 tensor'),
      );
      expect(
        () => pipeline.runSync(TensorInput(data: Uint8List(16), shape: [2, 2], payloadType: 'bogus')),
        throwsMorflow("Unknown payloadType 'bogus'"),
      );
      expect(
        () => pipeline.runSync(
          TensorInput(data: Uint8List(48), shape: [2, 2, 3], payloadType: 'image', colorSpace: 'cmyk'),
        ),
        throwsMorflow("Unknown color space 'cmyk'"),
      );
      expect(
        () => pipeline.runSync(TensorInput(data: Uint8List(16), shape: [2, 2], payloadType: 'image')),
        throwsMorflow('could not infer a color space'),
      );
      expect(
        () =>
            pipeline.runSync(TensorInput(data: Uint8List(16), shape: [2, 2], payloadType: 'audio', layout: 'diagonal')),
        throwsMorflow('Unknown audio layout'),
      );
    });

    test('unsupported Dart values are ArgumentErrors', () {
      final pipeline = morflow.fromStr(echo);
      expect(() => pipeline.runSync(Uint16List(2)), throwsArgumentError);
      expect(() => pipeline.runSync({'data': 1}), throwsArgumentError);
      expect(() => pipeline.runSync(TensorInput(data: [1, 2], shape: [2])), throwsArgumentError);
      expect(() => pipeline.runSync(TensorInput(data: Float32List(2), shape: [-2])), throwsArgumentError);
      expect(() => pipeline.runSync(3), throwsArgumentError);
      expect(() => pipeline.runSyncArgs([null]), throwsArgumentError);
    });

    test('async runs report errors as failed futures', () async {
      final pipeline = morflow.fromStr(echo);
      final dtype = pipeline.run(TensorInput(data: Uint8List(8), shape: [2], dtype: 'f16'));
      await expectLater(dtype, throwsMorflow('Unknown dtype'));
      final dart = pipeline.run(Uint16List(2));
      await expectLater(dart, throwsArgumentError);
    });
  });

  group('pipelines', () {
    test('parameters are listed in declaration order', () {
      final pipeline = morflow.fromStr('accept Audio \$audio_in\naccept IntArg \$rate = 44100\n\$audio_in >> emit');
      expect(pipeline.params, ['audio_in', 'rate']);
      expect(() => pipeline.params.add('x'), throwsUnsupportedError);
    });

    test('load compiles a file and reports a missing one', () {
      final directory = Directory.systemTemp.createTempSync('morflow-load-');
      try {
        final file = File('${directory.path}/echo.morf')..writeAsStringSync(echo);
        final pipeline = morflow.load(file.path);
        expect(pipeline.params, ['data']);
        expect(pipeline.runSync<MorflowTensor>(Float32List.fromList([4])).toFloat32Array(), [4]);
      } finally {
        directory.deleteSync(recursive: true);
      }
      expect(() => morflow.load('/definitely/missing/pipeline.morf'), throwsMorflow('IO error'));
    });

    test('fromString is an alias of fromStr', () {
      expect(morflow.fromString(echo).params, ['data']);
    });

    test('undeclared actions report their imports', () {
      expect(
        () => morflow.fromStr('accept Tensor \$data\n\$data >> identity >> emit'),
        throwsMorflow('not declared by the imports'),
      );
    });

    test('close releases the pipeline once and blocks further runs', () async {
      final pipeline = morflow.fromStr(echo);
      final running = pipeline.run<MorflowTensor>(Float32List.fromList([5]));
      pipeline.close();
      pipeline.close();
      expect((await running).toFloat32Array(), [5]);
      expect(() => pipeline.runSync(Float32List(1)), throwsStateError);
      await expectLater(pipeline.run(Float32List(1)), throwsStateError);
      expect(pipeline.params, ['data']);
    });

    test('many concurrent runs share one pipeline', () async {
      final pipeline = morflow.fromStr(echo);
      final outputs = await Future.wait([
        for (var i = 0; i < 100; i++) pipeline.run<MorflowTensor>(Float32List.fromList([i.toDouble()])),
      ]);
      for (var i = 0; i < 100; i++) {
        expect(outputs[i].toFloat32Array(), [i.toDouble()]);
      }
    });

    test('pipelines work in other isolates', () async {
      final values = await Isolate.run(() async {
        final pipeline = morflow.fromStr(echo);
        final output = await pipeline.run<MorflowTensor>(Float32List.fromList([1, 2]));
        return output.toFloat32Array().toList();
      });
      expect(values, [1, 2]);
    });

    test('toString lists the parameters', () {
      expect(morflow.fromStr(echo).toString(), 'Pipeline(params: [data])');
    });
  });

  group('versions', () {
    test('package, pubspec and native library agree', () {
      final pubspec = File('pubspec.yaml').readAsStringSync();
      final version = RegExp(r'^version: (.+)$', multiLine: true).firstMatch(pubspec)!.group(1);
      // The engine version, or the engine version plus a build number.
      expect(version, matches(RegExp('^${RegExp.escape(engineVersion)}(\\+\\d+)?\$')));
      // Compiling checks the native library's version against engineVersion.
      expect(() => morflow.fromStr(echo), returnsNormally);
    });
  });
}
