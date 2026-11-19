// The build hook (hook/build.dart) end to end: which engine and pack
// libraries it gives a build, from a release directory laid out like the
// GitHub releases. Each run is its own process (test/support/hook_runner.dart)
// because a failing hook exits the process.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:ffi' show Abi;
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:morflow/src/version.dart';
import 'package:test/test.dart';

import '../hook/release.dart';

const platforms = [
  'android-aarch64',
  'android-x86_64',
  'ios-arm64',
  'ios-arm64-sim',
  'linux-x86_64',
  'linux-aarch64',
  'darwin-arm64',
  'darwin-x86_64',
  'windows-x86_64',
];

late Directory releases;

/// Writes a release with [files] (name -> content) and its checksums.txt.
void release(String tag, Map<String, String> files) {
  final dir = Directory('${releases.path}/$tag')..createSync(recursive: true);
  final sums = StringBuffer();
  for (final MapEntry(key: name, value: content) in files.entries) {
    File('${dir.path}/$name').writeAsStringSync(content);
    sums.writeln('${sha256.convert(utf8.encode(content))}  $name');
  }
  File('${dir.path}/checksums.txt').writeAsStringSync(sums.toString());
}

class HookRun {
  HookRun(this.result);
  final ProcessResult result;
  String get log => '${result.stdout}\n${result.stderr}';
  bool get ok => result.exitCode == 0;

  /// Asset id -> {file, content}.
  Map<String, Map<String, Object?>> get assets {
    final line = (result.stdout as String).split('\n').firstWhere((l) => l.startsWith('ASSETS '), orElse: () => '');
    if (line.isEmpty) fail('The hook failed:\n$log');
    return (jsonDecode(line.substring(7)) as Map).map((k, v) => MapEntry(k as String, (v as Map).cast()));
  }
}

Future<HookRun> runHook({
  required String os,
  required String arch,
  bool simulator = false,
  Map<String, Object?>? defines,
}) async {
  final request = jsonEncode({
    'os': os,
    'arch': arch,
    'simulator': simulator,
    'defines': {'release_url': releases.uri.toString(), ...?defines},
  });
  return HookRun(await Process.run(Platform.resolvedExecutable, ['test/support/hook_runner.dart', request]));
}

const engine = 'package:morflow/src/bindings.dart';
String pack(String name) => 'package:morflow/src/packs/$name';

void main() {
  setUpAll(() {
    releases = Directory.systemTemp.createTempSync('morflow_releases');
    release(engineTag(packageVersion), {for (final p in platforms) capiAsset(p): 'engine $p'});
    release(packTag('base', '0.3.4'), {for (final p in platforms) packAsset('base', '0.3.4', p): 'base 0.3.4 $p'});
    release(packTag('audio_basics', '0.2.4'), {
      for (final p in platforms) packAsset('audio_basics', '0.2.4', p): 'audio_basics 0.2.4 $p',
    });
  });
  tearDownAll(() => releases.deleteSync(recursive: true));

  test('Android bundles the engine and the listed packs under the names it loads them by', () async {
    final run = await runHook(
      os: 'android',
      arch: 'arm64',
      defines: {
        'packs': {'base': '0.3.4', 'audio_basics': '0.2.4'},
      },
    );
    expect(run.assets, {
      engine: {'file': 'libmorflow_capi.so', 'content': 'engine android-aarch64'},
      pack('base'): {'file': 'libmorflow_pack_base.so', 'content': 'base 0.3.4 android-aarch64'},
      pack('audio_basics'): {
        'file': 'libmorflow_pack_audio_basics.so',
        'content': 'audio_basics 0.2.4 android-aarch64',
      },
    });
  });

  test('Android x86_64 (the emulator) gets its own libraries', () async {
    final run = await runHook(
      os: 'android',
      arch: 'x64',
      defines: {
        'packs': {'base': '0.3.4'},
      },
    );
    expect(run.assets[engine]!['content'], 'engine android-x86_64');
    expect(run.assets[pack('base')]!['content'], 'base 0.3.4 android-x86_64');
  });

  test('iOS devices and simulators get different libraries', () async {
    final device = await runHook(
      os: 'ios',
      arch: 'arm64',
      defines: {
        'packs': {'base': '0.3.4'},
      },
    );
    expect(device.assets, {
      engine: {'file': 'libmorflow_capi.dylib', 'content': 'engine ios-arm64'},
      pack('base'): {'file': 'libmorflow_pack_base.dylib', 'content': 'base 0.3.4 ios-arm64'},
    });
    final simulator = await runHook(
      os: 'ios',
      arch: 'arm64',
      simulator: true,
      defines: {
        'packs': {'base': '0.3.4'},
      },
    );
    expect(simulator.assets[engine]!['content'], 'engine ios-arm64-sim');
    expect(simulator.assets[pack('base')]!['content'], 'base 0.3.4 ios-arm64-sim');
  });

  test('desktop apps bundle the listed packs too', () async {
    // Windows is never the host of a checkout build here, so the engine is
    // downloaded on every test platform.
    final run = await runHook(
      os: 'windows',
      arch: 'x64',
      defines: {
        'packs': {'base': '0.3.4'},
      },
    );
    expect(run.assets, {
      engine: {'file': 'morflow_capi.dll', 'content': 'engine windows-x86_64'},
      pack('base'): {'file': 'morflow_pack_base.dll', 'content': 'base 0.3.4 windows-x86_64'},
    });
  }, skip: Platform.isWindows ? 'Windows builds the engine from this checkout' : false);

  test('macOS and Linux arm64 get their own pack libraries', () async {
    for (final (os, platform, file) in [
      ('macos', 'darwin-arm64', 'libmorflow_pack_base.dylib'),
      ('linux', 'linux-aarch64', 'libmorflow_pack_base.so'),
    ]) {
      final run = await runHook(
        os: os,
        arch: 'arm64',
        defines: {
          'packs': {'base': '0.3.4'},
        },
      );
      expect(run.assets[pack('base')], {'file': file, 'content': 'base 0.3.4 $platform'});
    }
  }, skip: [Abi.macosArm64, Abi.linuxArm64].contains(Abi.current()) ? 'the host builds its own engine' : false);

  test('release_url may be a plain directory path', () async {
    final run = await runHook(os: 'android', arch: 'arm64', defines: {'release_url': releases.path});
    expect(run.assets.keys, [engine]);
  });

  test('prebuilt_dir packs combine with downloaded ones, but a pack cannot come from both', () async {
    final prebuilt = Directory.systemTemp.createTempSync('morflow_prebuilt');
    addTearDown(() => prebuilt.deleteSync(recursive: true));
    Directory('${prebuilt.path}/android-aarch64').createSync();
    File('${prebuilt.path}/android-aarch64/libmorflow_capi.so').writeAsStringSync('local engine');
    File('${prebuilt.path}/android-aarch64/libmorflow_pack_nn_basics.so').writeAsStringSync('local nn_basics');

    final run = await runHook(
      os: 'android',
      arch: 'arm64',
      defines: {
        'prebuilt_dir': prebuilt.path,
        'packs': {'base': '0.3.4'},
      },
    );
    expect(run.assets, {
      engine: {'file': 'libmorflow_capi.so', 'content': 'local engine'},
      pack('nn_basics'): {'file': 'libmorflow_pack_nn_basics.so', 'content': 'local nn_basics'},
      pack('base'): {'file': 'libmorflow_pack_base.so', 'content': 'base 0.3.4 android-aarch64'},
    });

    File('${prebuilt.path}/android-aarch64/libmorflow_pack_base.so').writeAsStringSync('local base');
    final both = await runHook(
      os: 'android',
      arch: 'arm64',
      defines: {
        'prebuilt_dir': prebuilt.path,
        'packs': {'base': '0.3.4'},
      },
    );
    expect(both.ok, isFalse);
    expect(both.log, contains('base is both in hooks.user_defines.morflow.packs and in prebuilt_dir'));
  });

  group('fails the build', () {
    test('for a pack version that was never released', () async {
      final run = await runHook(
        os: 'android',
        arch: 'arm64',
        defines: {
          'packs': {'base': '9.9.9'},
        },
      );
      expect(run.ok, isFalse);
      expect(run.log, contains('Could not find release action_packs/base/v9.9.9'));
    });

    test('for a release without this platform', () async {
      release(packTag('desktop_only', '1.0.0'), {'other.txt': 'x'});
      final run = await runHook(
        os: 'android',
        arch: 'arm64',
        defines: {
          'packs': {'desktop_only': '1.0.0'},
        },
      );
      expect(run.ok, isFalse);
      expect(run.log, contains('Morflow release action_packs/desktop_only/v1.0.0 has no morflow_pack_desktop_only'));
    });

    test('for a file that does not match its checksum', () async {
      release(packTag('tampered', '1.0.0'), {packAsset('tampered', '1.0.0', 'android-aarch64'): 'original'});
      File(
        '${releases.path}/${packTag('tampered', '1.0.0')}/${packAsset('tampered', '1.0.0', 'android-aarch64')}',
      ).writeAsStringSync('changed');
      final run = await runHook(
        os: 'android',
        arch: 'arm64',
        defines: {
          'packs': {'tampered': '1.0.0'},
        },
      );
      expect(run.ok, isFalse);
      expect(run.log, contains('does not match its checksum'));
    });

    test('for packs that are not exact versions', () async {
      for (final (packs, message) in [
        ({'base': 'latest'}, 'must be an exact version such as 0.3.4'),
        (['base'], 'must map pack names to versions'),
        ({'Base!': '0.3.4'}, 'invalid pack name: Base!'),
      ]) {
        final run = await runHook(os: 'android', arch: 'arm64', defines: {'packs': packs});
        expect(run.ok, isFalse, reason: '$packs');
        expect(run.log, contains(message));
      }
    });

    test('when the engine release is missing', () async {
      final empty = Directory.systemTemp.createTempSync('morflow_empty');
      addTearDown(() => empty.deleteSync(recursive: true));
      final run = await runHook(os: 'ios', arch: 'arm64', defines: {'release_url': empty.uri.toString()});
      expect(run.ok, isFalse);
      expect(run.log, contains('Could not find release ${engineTag(packageVersion)}'));
    });
  });
}
