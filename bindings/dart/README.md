# Morflow for Flutter and Dart

Process images, audio and tensors in your Flutter app with Morflow pipelines. You write a pipeline once in a `.morf` file, and it runs inside your app on Android, iOS, macOS, Windows and Linux. It works offline, and your users don't need to install anything.

## Get started

**1. Add Morflow to your app**

```sh
flutter pub add morflow
```

**2. Write a pipeline**

Save it anywhere in your app's folder, such as `pipelines/boost.morf`. This one doubles the volume of a recording and returns it as a WAV file:

```
import audio_basics/latest

accept Audio $audio
$audio >> gain(linear=2.0) >> to_wav >> emit
```

**3. Get your app ready**

Install the Morflow CLI from [morflow.org](https://morflow.org/doc#cli-installation). Then run this in your app's folder, listing every pipeline your app uses:

```sh
morflow prep --prod-target flutter pipelines/boost.morf

# Or every pipeline in a folder
morflow prep --prod-target flutter pipelines
```

It shows what it will add to your `pubspec.yaml` and asks before changing anything. Press Enter to accept.

**4. Run the pipeline**

```dart
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:morflow/morflow.dart' as morflow;

Future<Uint8List> boost(Float32List samples) async {
  final pipeline = morflow.fromStr(await rootBundle.loadString('pipelines/boost.morf'));

  final wav = await pipeline.run<morflow.MorflowTensor>(
    morflow.TensorInput(data: samples, shape: [1, samples.length], payloadType: 'audio', sampleRate: 44100),
  );

  pipeline.close();
  return wav.toBuffer();
}
```

**5. Build as usual**

`flutter run` and `flutter build` include everything your pipelines need. When you add or change a pipeline, run step 3 again.

## Learn more

- [The Flutter guide](https://morflow.org/doc#exec-flutter): every step in detail, inputs and outputs, and how it works
- [Writing pipelines](https://morflow.org/doc#syntax)
- [Data types: tensors, images, audio and more](https://morflow.org/doc#types)
- [Questions and issues on GitHub](https://github.com/JiraPit/Morflow/issues)
