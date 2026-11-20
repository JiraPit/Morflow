# Morflow Showcase for Flutter

A small app that shows Morflow at work inside Flutter, on your phone or your computer:

- **Images:** blur, outline or brighten a picture, and see how many megapixels per second it handles.
- **Audio:** clean up a five-second recording and see how much faster than real time it runs.
- **Math:** multiply two big matrices, and compare Morflow with the same work in plain Dart.

Every demo has a **Run** button and a **Benchmark ×20** button that shows the median, best and first run.

## Run it

You only need Flutter. In this folder:

```sh
flutter run
```

Pick a device when Flutter asks: an Android phone or emulator, an iPhone or the iOS simulator, or your Mac, Windows or Linux desktop. The first build downloads Morflow's engine and the action packs the app uses, so it takes a little longer than usual.

## How it's made

- `pipelines/` holds the Morflow pipelines the app runs, such as `blur.morf` and `clean_audio.morf`. Change one and run the app again to see the difference.
- `lib/demos.dart` runs them, and `lib/main.dart` is the app around them.
- The `morflow` package and the packs in `pubspec.yaml` were set up with two commands:

  ```sh
  flutter pub add morflow
  morflow prep --prod-target flutter pipelines/*.morf
  ```

To use Morflow in your own app, follow the [Flutter guide](https://morflow.org/doc#exec-flutter).
