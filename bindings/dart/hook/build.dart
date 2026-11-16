// Provides the Morflow native library (bindings/capi) to Dart and Flutter
// builds. Applications never call it directly.
//
// The library comes from, in order:
//   1. The `prebuilt_dir` user-define: a directory with one subdirectory per
//      platform (android-aarch64, ios-arm64, linux-x86_64, …) holding
//      libmorflow_capi and any pack libraries (libmorflow_pack_<pack>) to
//      bundle into the app, which is how Android and iOS apps get actions:
//        hooks:
//          user_defines:
//            morflow:
//              prebuilt_dir: path/to/prebuilt
//   2. The `native_lib` user-define, a path to a library you built yourself:
//        hooks:
//          user_defines:
//            morflow:
//              native_lib: path/to/libmorflow_capi.so
//   3. A Morflow repository checkout: the library is built with cargo.
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

/// The asset id the Dart declarations in lib/src/bindings.dart load from.
const _assetName = 'src/bindings.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final code = input.config.code;
    final prebuilt = _prebuiltLibraries(input, output, code);
    final library =
        prebuilt?.engine ?? _configuredLibrary(input, output) ?? await _buildFromCheckout(input, output, code);
    output.assets.code.add(
      CodeAsset(package: input.packageName, name: _assetName, linkMode: DynamicLoadingBundled(), file: library),
    );
    // Pack libraries are loaded by the engine, not from Dart, but must ship
    // inside the app like the engine itself.
    for (final MapEntry(key: pack, value: file) in (prebuilt?.packs ?? const <String, Uri>{}).entries) {
      output.assets.code.add(
        CodeAsset(package: input.packageName, name: 'src/packs/$pack', linkMode: DynamicLoadingBundled(), file: file),
      );
    }
  });
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

Future<Uri> _buildFromCheckout(BuildInput input, BuildOutputBuilder output, CodeConfig code) async {
  final repo = input.packageRoot.resolve('../../');
  final crate = Directory.fromUri(repo.resolve('bindings/capi/'));
  if (!crate.existsSync()) {
    throw BuildError(
      message:
          'No Morflow native library is available for this build. Set '
          'hooks.user_defines.morflow.native_lib in your pubspec.yaml to a library built from bindings/capi.',
    );
  }
  if (code.targetOS != OS.current || code.targetArchitecture != Architecture.current) {
    throw BuildError(
      message:
          'Building the Morflow native library from source only supports the host platform '
          '(${OS.current}/${Architecture.current}), not ${code.targetOS}/${code.targetArchitecture}. '
          'Set hooks.user_defines.morflow.native_lib to a library built for the target.',
    );
  }

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
