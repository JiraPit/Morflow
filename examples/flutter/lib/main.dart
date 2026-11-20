// Morflow Showcase: runs the pipelines in pipelines/ inside a Flutter app and
// shows what they produce and how fast they run.
//
//   flutter run
//
// Every pipeline runs on a background thread through the morflow package; the
// packs they need are listed in pubspec.yaml by
// `morflow prep --prod-target flutter pipelines/*.morf`.
import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
// package:material_ui replaces this on Flutter 3.47 and later; this import
// keeps the app running on older Flutter versions too.
// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:morflow/morflow.dart' as morflow;

import 'demos.dart';

void main() => runApp(const ShowcaseApp());

class ShowcaseApp extends StatelessWidget {
  const ShowcaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFFE8613A);
    return MaterialApp(
      title: 'Morflow Showcase',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: seed), useMaterial3: true),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: const ShowcaseHome(),
    );
  }
}

/// Loads every pipeline once, from the app's assets.
Future<Map<String, morflow.Pipeline>> loadPipelines() async {
  return {for (final name in pipelineNames) name: morflow.fromStr(await rootBundle.loadString('pipelines/$name.morf'))};
}

class ShowcaseHome extends StatefulWidget {
  const ShowcaseHome({super.key});

  @override
  State<ShowcaseHome> createState() => _ShowcaseHomeState();
}

class _ShowcaseHomeState extends State<ShowcaseHome> {
  late final Future<Map<String, morflow.Pipeline>> _pipelines = loadPipelines();
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Morflow Showcase')),
      body: FutureBuilder(
        future: _pipelines,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(
              child: Padding(padding: const EdgeInsets.all(24), child: Text('${snapshot.error}')),
            );
          }
          final pipelines = snapshot.data;
          if (pipelines == null) return const Center(child: CircularProgressIndicator());
          return IndexedStack(
            index: _tab,
            children: [
              ImageDemo(pipelines: pipelines),
              AudioDemo(pipeline: pipelines['clean_audio']!),
              MathDemo(pipeline: pipelines['matmul']!),
            ],
          );
        },
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (tab) => setState(() => _tab = tab),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.image_outlined), label: 'Images'),
          NavigationDestination(icon: Icon(Icons.graphic_eq), label: 'Audio'),
          NavigationDestination(icon: Icon(Icons.grid_on), label: 'Math'),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Shared pieces

String ms(double value) => value < 10 ? '${value.toStringAsFixed(2)} ms' : '${value.toStringAsFixed(1)} ms';

/// The run and benchmark buttons and the latest timings.
class RunPanel extends StatelessWidget {
  const RunPanel({
    super.key,
    required this.busy,
    required this.onRun,
    required this.onBenchmark,
    required this.timings,
    required this.throughput,
    this.extra,
  });

  final bool busy;
  final VoidCallback onRun;
  final VoidCallback onBenchmark;
  final Timings? timings;

  /// What the median run means, such as "31 megapixels/s".
  final String Function(double medianMs) throughput;
  final Widget? extra;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final t = timings;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 12,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                FilledButton.icon(
                  onPressed: busy ? null : onRun,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Run'),
                ),
                OutlinedButton.icon(
                  onPressed: busy ? null : onBenchmark,
                  icon: const Icon(Icons.speed),
                  label: Text('Benchmark ×$benchmarkRuns'),
                ),
                if (busy) const SizedBox.square(dimension: 20, child: CircularProgressIndicator(strokeWidth: 2)),
              ],
            ),
            if (t != null) ...[
              const SizedBox(height: 16),
              Text(throughput(t.median), style: text.headlineSmall),
              const SizedBox(height: 4),
              Text(
                t.runs.length == 1
                    ? ms(t.first)
                    : 'median ${ms(t.median)} · best ${ms(t.best)} · first ${ms(t.first)} · ${t.runs.length} runs',
                style: text.bodyMedium,
              ),
            ],
            if (extra != null) ...[const SizedBox(height: 12), extra!],
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Images

/// A colorful test picture, drawn with Flutter, as RGB bytes.
Future<Uint8List> drawTestImage(int size) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder);
  final s = size.toDouble();
  canvas.drawRect(
    Rect.fromLTWH(0, 0, s, s),
    Paint()
      ..shader = ui.Gradient.linear(
        Offset.zero,
        Offset(s, s),
        const [Color(0xFF1B2A49), Color(0xFFE8613A), Color(0xFFF4D35E)],
        const [0, 0.6, 1],
      ),
  );
  final random = math.Random(7);
  for (var i = 0; i < 28; i++) {
    final color = HSVColor.fromAHSV(0.85, random.nextDouble() * 360, 0.7, 0.95).toColor();
    final center = Offset(random.nextDouble() * s, random.nextDouble() * s);
    canvas.drawCircle(center, s * (0.03 + random.nextDouble() * 0.09), Paint()..color = color);
  }
  final stripe = Paint()
    ..color = const Color(0x55FFFFFF)
    ..strokeWidth = 2;
  for (var y = 0.0; y < s; y += s / 18) {
    canvas.drawLine(Offset(0, y), Offset(s, y + s / 6), stripe);
  }
  final label = TextPainter(
    text: TextSpan(
      text: 'Morflow',
      style: TextStyle(fontSize: s / 7, fontWeight: FontWeight.w800, color: Colors.white),
    ),
    textDirection: TextDirection.ltr,
  )..layout();
  label.paint(canvas, Offset((s - label.width) / 2, (s - label.height) / 2));
  final image = await recorder.endRecording().toImage(size, size);
  final rgba = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!.buffer.asUint8List();
  return rgbaToRgb(rgba);
}

/// An image tensor ([height, width, channels], u8) as something Flutter can draw.
Future<ui.Image> toUiImage(Uint8List pixels, int width, int height, int channels) {
  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(toRgba(pixels, channels), width, height, ui.PixelFormat.rgba8888, completer.complete);
  return completer.future;
}

class ImageDemo extends StatefulWidget {
  const ImageDemo({super.key, required this.pipelines});
  final Map<String, morflow.Pipeline> pipelines;

  @override
  State<ImageDemo> createState() => _ImageDemoState();
}

class _ImageDemoState extends State<ImageDemo> {
  static const _effects = {'blur': 'Blur', 'edges': 'Edges', 'vivid': 'Vivid'};

  String _effect = 'blur';
  Uint8List? _source;
  ui.Image? _original;
  ui.Image? _result;
  Timings? _timings;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    drawTestImage(imageSize).then((rgb) async {
      final original = await toUiImage(rgb, imageSize, imageSize, 3);
      if (!mounted) return;
      setState(() {
        _source = rgb;
        _original = original;
      });
    });
  }

  Future<void> _run(int count) async {
    final source = _source;
    if (source == null) return;
    setState(() => _busy = true);
    final run = await runImage(widget.pipelines[_effect]!, source, count);
    final [height, width, channels] = run.output.shape;
    final result = await toUiImage(run.output.data, width, height, channels);
    if (!mounted) return;
    setState(() {
      _result = result;
      _timings = run.timings;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Image effects', style: text.titleLarge),
        const SizedBox(height: 4),
        Text('A $imageSize×$imageSize picture through pipelines/$_effect.morf.', style: text.bodyMedium),
        const SizedBox(height: 12),
        SegmentedButton<String>(
          segments: [for (final e in _effects.entries) ButtonSegment(value: e.key, label: Text(e.value))],
          selected: {_effect},
          onSelectionChanged: _busy
              ? null
              : (s) => setState(() {
                  _effect = s.first;
                  _result = null;
                  _timings = null;
                }),
        ),
        const SizedBox(height: 12),
        RunPanel(
          busy: _busy,
          onRun: () => _run(1),
          onBenchmark: () => _run(benchmarkRuns),
          timings: _timings,
          throughput: (median) => '${megapixelsPerSecond(median).toStringAsFixed(0)} megapixels/s',
        ),
        const SizedBox(height: 12),
        LayoutBuilder(
          builder: (context, constraints) {
            final panes = [
              _Picture(title: 'Original', image: _original),
              _Picture(title: _effects[_effect]!, image: _result, placeholder: 'Press Run'),
            ];
            return constraints.maxWidth > 640
                ? Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [for (final p in panes) Expanded(child: p)],
                  )
                : Column(children: panes);
          },
        ),
      ],
    );
  }
}

class _Picture extends StatelessWidget {
  const _Picture({required this.title, required this.image, this.placeholder});
  final String title;
  final ui.Image? image;
  final String? placeholder;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 6),
          AspectRatio(
            aspectRatio: 1,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: image != null
                  ? RawImage(image: image, fit: BoxFit.cover)
                  : ColoredBox(
                      color: Theme.of(context).colorScheme.surfaceContainerHighest,
                      child: Center(child: Text(placeholder ?? '')),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Audio

class AudioDemo extends StatefulWidget {
  const AudioDemo({super.key, required this.pipeline});
  final morflow.Pipeline pipeline;

  @override
  State<AudioDemo> createState() => _AudioDemoState();
}

class _AudioDemoState extends State<AudioDemo> {
  final Float32List _recording = makeRecording();
  Float32List? _cleaned;
  Timings? _timings;
  bool _busy = false;

  Future<void> _run(int count) async {
    setState(() => _busy = true);
    final run = await runAudio(widget.pipeline, _recording, count);
    if (!mounted) return;
    setState(() {
      _cleaned = run.output.toFloat32Array();
      _timings = run.timings;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Clean up a recording', style: text.titleLarge),
        const SizedBox(height: 4),
        Text(
          '$recordingSeconds seconds at 44.1 kHz: a low-pass filter takes out the hiss, a compressor evens out the loud '
          'and quiet parts, and normalize brings the peak to -1 dB (pipelines/clean_audio.morf).',
          style: text.bodyMedium,
        ),
        const SizedBox(height: 12),
        RunPanel(
          busy: _busy,
          onRun: () => _run(1),
          onBenchmark: () => _run(benchmarkRuns),
          timings: _timings,
          throughput: (median) => '${realTimeFactor(median).toStringAsFixed(0)}× faster than real time',
        ),
        const SizedBox(height: 12),
        _Waveform(title: 'Before', samples: _recording, color: scheme.outline),
        const SizedBox(height: 12),
        if (_cleaned case final cleaned?)
          _Waveform(title: 'After', samples: cleaned, color: scheme.primary)
        else
          const SizedBox(height: 120, child: Center(child: Text('Press Run'))),
      ],
    );
  }
}

class _Waveform extends StatelessWidget {
  const _Waveform({required this.title, required this.samples, required this.color});
  final String title;
  final Float32List samples;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 6),
        SizedBox(
          height: 120,
          width: double.infinity,
          child: CustomPaint(painter: _WaveformPainter(samples, color)),
        ),
      ],
    );
  }
}

class _WaveformPainter extends CustomPainter {
  _WaveformPainter(this.samples, this.color);
  final Float32List samples;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1;
    final columns = size.width.floor().clamp(1, samples.length);
    final perColumn = samples.length / columns;
    final middle = size.height / 2;
    for (var x = 0; x < columns; x++) {
      var low = 0.0, high = 0.0;
      final end = ((x + 1) * perColumn).floor();
      for (var i = (x * perColumn).floor(); i < end; i++) {
        low = math.min(low, samples[i]);
        high = math.max(high, samples[i]);
      }
      canvas.drawLine(Offset(x + 0.5, middle - high * middle), Offset(x + 0.5, middle - low * middle), paint);
    }
  }

  @override
  bool shouldRepaint(_WaveformPainter old) => old.samples != samples || old.color != color;
}

// ---------------------------------------------------------------------------
// Math

class MathDemo extends StatefulWidget {
  const MathDemo({super.key, required this.pipeline});
  final morflow.Pipeline pipeline;

  @override
  State<MathDemo> createState() => _MathDemoState();
}

class _MathDemoState extends State<MathDemo> {
  int _n = 512;
  Timings? _timings;
  double? _dartMs;
  bool? _checked;
  bool _busy = false;

  Future<void> _run(int count) async {
    setState(() {
      _busy = true;
      _dartMs = null;
    });
    final run = await runMatmul(widget.pipeline, _n, count);
    if (!mounted) return;
    setState(() {
      _timings = run.timings;
      _checked = run.checked;
      _busy = false;
    });
  }

  Future<void> _compare() async {
    setState(() => _busy = true);
    final a = randomMatrix(_n, 1), b = randomMatrix(_n, 2);
    final watch = Stopwatch()..start();
    await compute(dartMatmul, (a, b, _n));
    if (!mounted) return;
    setState(() {
      _dartMs = watch.elapsedMicroseconds / 1000;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final t = _timings;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Matrix multiplication', style: text.titleLarge),
        const SizedBox(height: 4),
        Text('Two random $_n×$_n float32 matrices multiplied by pipelines/matmul.morf.', style: text.bodyMedium),
        const SizedBox(height: 12),
        SegmentedButton<int>(
          segments: const [
            ButtonSegment(value: 256, label: Text('256')),
            ButtonSegment(value: 512, label: Text('512')),
            ButtonSegment(value: 1024, label: Text('1024')),
          ],
          selected: {_n},
          onSelectionChanged: _busy
              ? null
              : (s) => setState(() {
                  _n = s.first;
                  _timings = null;
                  _dartMs = null;
                  _checked = null;
                }),
        ),
        const SizedBox(height: 12),
        RunPanel(
          busy: _busy,
          onRun: () => _run(1),
          onBenchmark: () => _run(benchmarkRuns),
          timings: t,
          throughput: (median) => '${gflops(_n, median).toStringAsFixed(1)} GFLOP/s',
          extra: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (_checked case final checked?) Text(checked ? 'Result checked against Dart ✓' : 'Result differs ✗'),
              const SizedBox(height: 8),
              TextButton.icon(
                onPressed: _busy ? null : _compare,
                icon: const Icon(Icons.compare_arrows),
                label: const Text('Compare with plain Dart'),
              ),
              if (_dartMs case final dart?)
                Text(
                  'Plain Dart: ${ms(dart)}'
                  '${t != null ? ' · Morflow is ${(dart / t.median).toStringAsFixed(1)}× faster' : ''}',
                ),
            ],
          ),
        ),
      ],
    );
  }
}
