// Downloads Morflow's native libraries from its GitHub releases for the build
// hook (hook/build.dart). It lives here because pub.dev accepts only
// hook/build.dart in hook/. Not part of the package's API.
//   - the C interface library, from the engine release v<version>;
//   - pack libraries (every action of a pack in one library) that apps bundle,
//     from the pack's release action_packs/<pack>/v<version>.
// Every file is checked against the release's checksums.txt and kept in a
// cache, so later builds work offline.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

const defaultReleaseUrl = 'https://github.com/JiraPit/Morflow/releases/download';

/// A release or one of its files could not be used.
class ReleaseException implements Exception {
  ReleaseException(this.message);
  final String message;
  @override
  String toString() => message;
}

String extension(String platform) => platform.startsWith('windows-')
    ? 'dll'
    : platform.startsWith('darwin-') || platform.startsWith('ios-')
    ? 'dylib'
    : 'so';

/// The file name the operating system loads, such as `libmorflow_capi.so`.
String libraryFile(String stem, String platform) =>
    '${platform.startsWith('windows-') ? '' : 'lib'}$stem.${extension(platform)}';

String engineTag(String version) => 'v$version';
String packTag(String pack, String version) => 'action_packs/$pack/v$version';
String capiAsset(String platform) => 'libmorflow_capi-$platform.${extension(platform)}';
String packAsset(String pack, String version, String platform) =>
    'morflow_pack_$pack-$version-$platform.${extension(platform)}';

final _packName = RegExp(r'^[a-z0-9_]+$');
final _version = RegExp(r'^\d+\.\d+\.\d+$');

/// The `packs` user-define: pack name -> exact version.
Map<String, String> parsePacks(Object? define) {
  if (define == null) return const {};
  if (define is! Map) {
    throw ReleaseException(
      'hooks.user_defines.morflow.packs must map pack names to versions, such as:\n'
      '  packs:\n'
      '    base: 0.3.4\n'
      '    audio_basics: 0.2.4',
    );
  }
  final packs = <String, String>{};
  for (final MapEntry(:key, :value) in define.entries) {
    if (key is! String || !_packName.hasMatch(key)) {
      throw ReleaseException('hooks.user_defines.morflow.packs has an invalid pack name: $key');
    }
    if (value is! String || !_version.hasMatch(value)) {
      throw ReleaseException(
        'hooks.user_defines.morflow.packs.$key must be an exact version such as 0.3.4 (in quotes if YAML '
        'reads it as a number), not $value.',
      );
    }
    packs[key] = value;
  }
  return packs;
}

/// Parses a `checksums.txt` written by `sha256sum`.
Map<String, String> parseChecksums(String text) => {
  for (final line in const LineSplitter().convert(text))
    if (RegExp(r'^([0-9a-f]{64}) [ *]?(.+)$').firstMatch(line.trim()) case final match?)
      match.group(2)!: match.group(1)!,
};

/// Downloads release files into [cache].
class ReleaseDownloader {
  ReleaseDownloader({required this.baseUrl, required this.cache, HttpClient? client})
    : _client = client ?? HttpClient();

  /// Where releases are, `<baseUrl>/<tag>/<asset>`: GitHub's download URL,
  /// a mirror, or a `file:` directory with the same layout.
  final Uri baseUrl;
  final Directory cache;
  final HttpClient _client;

  /// Downloads [asset] from release [tag] and returns it saved as [saveAs],
  /// the name the operating system loads it by. A cached copy is reused while
  /// it matches the release's checksum.
  Future<File> fetch({required String tag, required String asset, required String saveAs}) async {
    final dir = Directory.fromUri(cache.uri.resolve('${_safe(tag)}/${_safe(asset)}/'));
    final file = File.fromUri(dir.uri.resolve(saveAs));
    final checksumsFile = File.fromUri(cache.uri.resolve('${_safe(tag)}/checksums.txt'));

    // Published files never change, so a cached checksum stays valid.
    if (checksumsFile.existsSync() && file.existsSync()) {
      final expected = parseChecksums(checksumsFile.readAsStringSync())[asset];
      if (expected != null && await _sha256(file) == expected) return file;
    }

    final checksumsText = utf8.decode(await _download(tag, 'checksums.txt', what: 'release $tag'));
    final expected = parseChecksums(checksumsText)[asset];
    if (expected == null) {
      throw ReleaseException('Morflow release $tag has no $asset (${_url(tag, '')}).');
    }
    checksumsFile.parent.createSync(recursive: true);
    checksumsFile.writeAsStringSync(checksumsText);
    if (file.existsSync() && await _sha256(file) == expected) return file;

    final bytes = await _download(tag, asset, what: '$asset from release $tag');
    final actual = sha256.convert(bytes).toString();
    if (actual != expected) {
      throw ReleaseException(
        '$asset from Morflow release $tag does not match its checksum (expected $expected, got $actual). '
        'The download was corrupted or changed; it was not used.',
      );
    }
    dir.createSync(recursive: true);
    final partial = File('${file.path}.partial')..writeAsBytesSync(bytes, flush: true);
    partial.renameSync(file.path);
    return file;
  }

  void close() => _client.close();

  Uri _url(String tag, String asset) {
    final base = baseUrl.path.endsWith('/') ? baseUrl : baseUrl.replace(path: '${baseUrl.path}/');
    return base.resolve('$tag/$asset');
  }

  Future<List<int>> _download(String tag, String asset, {required String what}) async {
    final url = _url(tag, asset);
    if (url.scheme == 'file') {
      final file = File.fromUri(url);
      if (!file.existsSync()) throw ReleaseException('Could not find $what: ${file.path} does not exist.');
      return file.readAsBytesSync();
    }
    try {
      final request = await _client.getUrl(url);
      final response = await request.close();
      final bytes = await response.fold<BytesBuilder>(BytesBuilder(copy: false), (b, chunk) => b..add(chunk));
      if (response.statusCode == HttpStatus.notFound) {
        throw ReleaseException('Could not find $what: $url returned 404. Is the version published?');
      }
      if (response.statusCode != HttpStatus.ok) {
        throw ReleaseException('Could not download $what: $url returned HTTP ${response.statusCode}.');
      }
      return bytes.takeBytes();
    } on IOException catch (error) {
      throw ReleaseException('Could not download $what from $url: $error');
    }
  }

  static Future<String> _sha256(File file) async => (await sha256.bind(file.openRead()).first).toString();

  // Tags such as action_packs/base/v0.3.4 become one directory name.
  static String _safe(String name) => name.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
}
