import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:morflow/morflow.dart' as morflow;

void main() => runApp(const MorflowExample());

/// Runs two pipelines with actions bundled into the app and shows the results.
class MorflowExample extends StatefulWidget {
  const MorflowExample({super.key});

  @override
  State<MorflowExample> createState() => _MorflowExampleState();
}

class _MorflowExampleState extends State<MorflowExample> {
  final _lines = <String>[];

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    try {
      final identity = morflow.fromStr(r'''
        accept Tensor $tensor
        $tensor >> base/latest/identity >> emit
      ''');
      final tensor = await identity.run<morflow.MorflowTensor>(Float32List.fromList([1, 2, 3]));
      _show('identity: ${tensor.toFloat32Array()}');

      final audio = morflow.fromStr(r'''
        import audio_basics/latest
        accept Audio $audio
        $audio >> gain(linear=2.0) >> to_wav >> emit
      ''');
      final wav = await audio.run<morflow.MorflowTensor>(
        morflow.TensorInput(
          data: Float32List(2 * 4410),
          shape: [2, 4410],
          payloadType: 'audio',
          sampleRate: 44100,
        ),
      );
      _show('audio: ${wav.toBuffer().length} byte WAV');
    } catch (error) {
      _show('error: $error');
    }
  }

  void _show(String line) => setState(() => _lines.add(line));

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Morflow',
    home: Scaffold(
      appBar: AppBar(title: const Text('Morflow')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [for (final line in _lines) Text(line, key: ValueKey(line))],
      ),
    ),
  );
}
