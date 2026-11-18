// Provides the Morflow native library (bindings/capi) to Dart and Flutter
// builds, and the action packs that Android and iOS apps bundle.
// Applications never call it directly.
//
// The engine library comes from, in order:
//   1. The `prebuilt_dir` user-define: a directory with one subdirectory per
//      platform (android-aarch64, ios-arm64, linux-x86_64, …) holding
//      libmorflow_capi and any pack libraries (libmorflow_pack_<pack>) to
//      bundle into the app.
//   2. The `native_lib` user-define, a path to a library you built yourself.
//   3. A Morflow repository checkout, for the host platform: built with cargo.
//   4. Otherwise, the engine's GitHub release for this package's version.
//
// Android and iOS apps cannot download actions at run time, so they bundle
// the action packs listed in the `packs` user-define, downloaded from each
// pack's GitHub release. Desktop programs use the action cache instead
// (`morflow prep`), so the list is ignored there:
//
//   hooks:
//     user_defines:
//       morflow:
//         packs:
//           base: 0.3.4
//           audio_basics: 0.2.4
//
// `release_url` replaces https://github.com/JiraPit/Morflow/releases/download
// with a mirror or a local directory using the same <tag>/<file> layout.
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:morflow/src/version.dart';

import 'release.dart';

/// The asset id the Dart declarations in lib/src/bindings.dart load from.
const _assetName = 'src/bindings.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    final platform = platformName(code);
    final mobile = code.targetOS == OS.android || code.targetOS == OS.iOS;
    final prebuilt = _prebuiltLibraries(input, output, code);
    final requested = _guard(() => parsePacks(input.userDefines['packs']));
    final downloader = ReleaseDownloader(
      baseUrl: _releaseUrl(input),
      cache: Directory.fromUri(input.outputDirectoryShared.resolve('releases/')),
    );
    try {
      final library =
          prebuilt?.engine ??
          _configuredLibrary(input, output) ??
          await _buildFromCheckout(input, output, code) ??
          await _guardAsync(
            () async => (await downloader.fetch(
              tag: engineTag(packageVersion),
              asset: capiAsset(platform),
              saveAs: libraryFile('morflow_capi', platform),
            )).uri,
          );
      output.assets.code.add(
        CodeAsset(package: input.packageName, name: _assetName, linkMode: DynamicLoadingBundled(), file: library),
      );

      // Pack libraries are loaded by the engine, not from Dart, but must ship
      // inside the app like the engine itself.
      final packs = {...?prebuilt?.packs};
      if (mobile) {
        for (final MapEntry(key: pack, value: version) in requested.entries) {
          if (packs.containsKey(pack)) {
            throw BuildError(
              message: '$pack is both in hooks.user_defines.morflow.packs and in prebuilt_dir; list it once.',
            );
          }
          packs[pack] = await _guardAsync(
            () async => (await downloader.fetch(
              tag: packTag(pack, version),
              asset: packAsset(pack, version, platform),
              saveAs: libraryFile('morflow_pack_$pack', platform),
            )).uri,
          );
        }
      }
      for (final MapEntry(key: pack, value: file) in packs.entries) {
        output.assets.code.add(
          CodeAsset(package: input.packageName, name: 'src/packs/$pack', linkMode: DynamicLoadingBundled(), file: file),
        );
      }
    } finally {
      downloader.close();
    }
  });
}

T _guard<T>(T Function() body) {
  try {
    return body();
  } on ReleaseException catch (error) {
    throw BuildError(message: error.message);
  }
}

Future<T> _guardAsync<T>(Future<T> Function() body) async {
  try {
    return await body();
  } on ReleaseException catch (error) {
    throw BuildError(message: error.message);
  }
}

Uri _releaseUrl(BuildInput input) {
  final define = input.userDefines['release_url'];
  if (define == null) return Uri.parse(defaultReleaseUrl);
  if (define is! String) throw BuildError(message: 'hooks.user_defines.morflow.release_url must be a string.');
  if (RegExp(r'^(https?|file):').hasMatch(define)) return Uri.parse(define);
  // A directory, relative to the pubspec that sets it.
  return Directory.fromUri(input.userDefines.path('release_url')!).uri;
}

/// The platform name used for artifacts, such as `android-aarch64`.
String platformName(CodeConfig code) {
  final arch = switch (code.targetArchitecture) {
    Architecture.arm64 => 'aarch64',
    Architecture.x64 => 'x86_64',
    final other => throw BuildError(message: 'Morflow does not support the $other architecture'),
  };
  return switch (code.targetOS) {
    OS.android => 'android-$arch',
    OS.iOS => code.iOS.targetSdk == IOSSdk.iPhoneSimulator ? 'ios-arm64-sim' : 'ios-arm64',
    OS.linux => 'linux-$arch',
    OS.macOS => arch == 'aarch64' ? 'darwin-arm64' : 'darwin-x86_64',
    OS.windows => 'windows-$arch',
    final other => throw BuildError(message: 'Morflow does not support $other'),
  };
}

/// The engine and pack libraries from the `prebuilt_dir` user-define.
({Uri engine, Map<String, Uri> packs})? _prebuiltLibraries(
  BuildInput input,
  BuildOutputBuilder output,
  CodeConfig code,
) {
  final root = input.userDefines.path('prebuilt_dir');
  if (root == null) return null;
  final platform = platformName(code);
  // Directory.uri ends in a slash, so the platform resolves inside it.
  final directory = Directory.fromUri(Directory.fromUri(root).uri.resolve('$platform/'));
  final engine = File.fromUri(directory.uri.resolve(libraryFileName(code.targetOS)));
  if (!engine.existsSync()) {
    throw BuildError(
      message:
          'hooks.user_defines.morflow.prebuilt_dir has no ${engine.uri.pathSegments.last} for $platform '
          '(looked in ${directory.path}).',
    );
  }
  output.dependencies.add(engine.uri);
  final packs = <String, Uri>{};
  final packFile = RegExp(r'^(?:lib)?morflow_pack_(\w+)\.(?:so|dylib|dll)$');
  for (final file in directory.listSync().whereType<File>()) {
    final match = packFile.firstMatch(file.uri.pathSegments.last);
    if (match == null) continue;
    packs[match.group(1)!] = file.uri;
    output.dependencies.add(file.uri);
  }
  return (engine: engine.uri, packs: packs);
}

/// The file name of the native library on [os].
String libraryFileName(OS os) => switch (os) {
  OS.windows => 'morflow_capi.dll',
  OS.macOS || OS.iOS => 'libmorflow_capi.dylib',
  _ => 'libmorflow_capi.so',
};

Uri? _configuredLibrary(BuildInput input, BuildOutputBuilder output) {
  final configured = input.userDefines.path('native_lib');
  if (configured == null) return null;
  final file = File.fromUri(configured);
  if (!file.existsSync()) {
    throw BuildError(message: 'hooks.user_defines.morflow.native_lib points to ${file.path}, which does not exist.');
  }
  output.dependencies.add(file.uri);
  return file.uri;
}

/// Builds the library with cargo inside a Morflow repository checkout, for
/// the host platform only. Null outside a checkout or for another platform.
Future<Uri?> _buildFromCheckout(BuildInput input, BuildOutputBuilder output, CodeConfig code) async {
  final repo = input.packageRoot.resolve('../../');
  final crate = Directory.fromUri(repo.resolve('bindings/capi/'));
  if (!crate.existsSync()) return null;
  if (code.targetOS != OS.current || code.targetArchitecture != Architecture.current) return null;

  final result = await Process.run(
    'cargo',
    ['build', '--release', '--locked', '-p', 'morflow-capi'],
    workingDirectory: Directory.fromUri(repo).path,
    runInShell: Platform.isWindows,
  );
  if (result.exitCode != 0) {
    throw BuildError(message: 'cargo build -p morflow-capi failed:\n${result.stdout}\n${result.stderr}');
  }

  // Rebuild when any Rust source the library is compiled from changes.
  for (final dir in ['bindings/capi', 'pipeline', 'core_types', 'parser']) {
    final root = Directory.fromUri(repo.resolve('$dir/'));
    for (final entity in root.listSync(recursive: true)) {
      if (entity is File && (entity.path.endsWith('.rs') || entity.path.endsWith('Cargo.toml'))) {
        output.dependencies.add(entity.uri);
      }
    }
  }
  output.dependencies.add(repo.resolve('Cargo.lock'));

  final library = File.fromUri(repo.resolve('target/release/${libraryFileName(code.targetOS)}'));
  if (!library.existsSync()) {
    throw BuildError(message: 'cargo finished but ${library.path} was not produced.');
  }
  return library.uri;
}
