// Runs the package's build hook (hook/build.dart) once for test/hook_test.dart
// and prints the code assets it produced as JSON. The hook exits the process
// when it fails, so each run gets its own process.
//
//   dart test/support/hook_runner.dart '{"os": "android", "arch": "arm64", "defines": {...}}'
import 'dart:convert';
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

import '../../hook/build.dart' as hook;

Future<void> main(List<String> args) async {
  final request = jsonDecode(args.single) as Map<String, Object?>;
  final os = OS.values.firstWhere((os) => os.name == request['os']);
  final arch = Architecture.values.firstWhere((arch) => arch.name == request['arch']);
  await testCodeBuildHook(
    mainMethod: hook.main,
    targetOS: os,
    targetArchitecture: arch,
    targetIOSSdk: request['simulator'] == true ? IOSSdk.iPhoneSimulator : IOSSdk.iPhoneOS,
    iOSDeploymentVersion: os == OS.iOS ? AppleVersion(13) : null,
    macOSDeploymentVersion: os == OS.macOS ? AppleVersion(13) : null,
    userDefines: PackageUserDefines(
      workspacePubspec: PackageUserDefinesSource(
        defines: (request['defines'] as Map<String, Object?>?) ?? const {},
        basePath: Directory.current.uri,
      ),
    ),
    check: (input, output) {
      // Each asset's file name and the start of its contents (test libraries
      // are short text files).
      final assets = {
        for (final asset in output.assets.code)
          asset.id: {
            'file': asset.file!.pathSegments.last,
            'content': utf8.decode(File.fromUri(asset.file!).readAsBytesSync().take(64).toList(), allowMalformed: true),
          },
      };
      stdout.writeln('ASSETS ${jsonEncode(assets)}');
    },
  );
}
