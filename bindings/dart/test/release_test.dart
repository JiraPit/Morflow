// Downloading native libraries from Morflow's releases (lib/src/hook/release.dart)
// over HTTP, as the build hook does from GitHub.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:test/test.dart';

import 'package:morflow/src/hook/release.dart';

const tag = 'action_packs/base/v0.3.4';
const asset = 'morflow_pack_base-0.3.4-android-aarch64.so';
const content = 'pack library bytes';

String sum(String text) => sha256.convert(utf8.encode(text)).toString();

void main() {
  late HttpServer server;
  late Directory cache;
  // Path -> (status, body); requests are recorded in order.
  late Map<String, (int, String)> routes;
  late List<String> requests;
  late int port;

  setUp(() async {
    routes = {
      '/download/$tag/checksums.txt': (200, '${sum(content)}  $asset\n${sum('other')} *other.so\n'),
      '/download/$tag/$asset': (302, '/storage/$asset'),
      '/storage/$asset': (200, content),
    };
    requests = [];
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    port = server.port;
    server.listen((request) {
      requests.add(request.uri.path);
      final (status, body) = routes[request.uri.path] ?? (404, 'Not Found');
      request.response.statusCode = status;
      if (status == 302) {
        request.response.headers.set('location', body);
      } else {
        request.response.write(body);
      }
      request.response.close();
    });
    cache = Directory.systemTemp.createTempSync('morflow_release_cache');
  });

  tearDown(() async {
    await server.close(force: true);
    cache.deleteSync(recursive: true);
  });

  ReleaseDownloader downloader() =>
      ReleaseDownloader(baseUrl: Uri.parse('http://127.0.0.1:$port/download'), cache: cache);

  Future<File> fetch() async {
    final d = downloader();
    try {
      return await d.fetch(tag: tag, asset: asset, saveAs: 'libmorflow_pack_base.so');
    } finally {
      d.close();
    }
  }

  test('downloads through redirects and saves the file under the name it is loaded by', () async {
    final file = await fetch();
    expect(file.uri.pathSegments.last, 'libmorflow_pack_base.so');
    expect(file.readAsStringSync(), content);
    expect(requests, ['/download/$tag/checksums.txt', '/download/$tag/$asset', '/storage/$asset']);
  });

  test('later builds use the cache without the network', () async {
    final first = await fetch();
    await server.close(force: true);
    final second = await fetch();
    expect(second.path, first.path);
    expect(second.readAsStringSync(), content);
  });

  test('a damaged cached file is downloaded again', () async {
    final file = await fetch();
    file.writeAsStringSync('damaged');
    requests.clear();
    expect((await fetch()).readAsStringSync(), content);
    expect(requests, contains('/storage/$asset'));
  });

  test('a download that does not match its checksum is rejected and not kept', () async {
    routes['/storage/$asset'] = (200, 'tampered');
    await expectLater(
      fetch(),
      throwsA(isA<ReleaseException>().having((e) => e.message, 'message', contains('does not match its checksum'))),
    );
    final kept = cache.listSync(recursive: true).whereType<File>().map((f) => f.uri.pathSegments.last);
    expect(kept, isNot(contains('libmorflow_pack_base.so')));
    expect(kept.where((name) => name.endsWith('.partial')), isEmpty);
  });

  test('a release without the file names the file', () async {
    routes['/download/$tag/checksums.txt'] = (200, '${sum('other')}  other.so\n');
    await expectLater(
      fetch(),
      throwsA(isA<ReleaseException>().having((e) => e.message, 'message', contains('has no $asset'))),
    );
  });

  test('a missing release says so', () async {
    routes.remove('/download/$tag/checksums.txt');
    await expectLater(
      fetch(),
      throwsA(
        isA<ReleaseException>().having(
          (e) => e.message,
          'message',
          allOf(contains('Could not find release $tag'), contains('404')),
        ),
      ),
    );
  });

  test('server errors and unreachable servers are reported', () async {
    routes['/storage/$asset'] = (500, 'oops');
    await expectLater(
      fetch(),
      throwsA(isA<ReleaseException>().having((e) => e.message, 'message', contains('HTTP 500'))),
    );
    await server.close(force: true);
    await expectLater(
      fetch(),
      throwsA(isA<ReleaseException>().having((e) => e.message, 'message', contains('Could not download'))),
    );
  });

  test('a base URL with a trailing slash works the same', () async {
    final d = ReleaseDownloader(baseUrl: Uri.parse('http://127.0.0.1:$port/download/'), cache: cache);
    addTearDown(d.close);
    expect((await d.fetch(tag: tag, asset: asset, saveAs: 'lib.so')).readAsStringSync(), content);
  });

  group('names', () {
    test('match the release assets CI publishes (scripts/native_targets.py)', () {
      expect(capiAsset('android-aarch64'), 'libmorflow_capi-android-aarch64.so');
      expect(capiAsset('ios-arm64-sim'), 'libmorflow_capi-ios-arm64-sim.dylib');
      expect(capiAsset('windows-x86_64'), 'libmorflow_capi-windows-x86_64.dll');
      expect(packAsset('base', '0.3.4', 'ios-arm64'), 'morflow_pack_base-0.3.4-ios-arm64.dylib');
      expect(engineTag('0.3.2'), 'v0.3.2');
      expect(packTag('audio_basics', '0.2.4'), 'action_packs/audio_basics/v0.2.4');
    });

    test('match what the engine loads', () {
      expect(libraryFile('morflow_pack_base', 'android-x86_64'), 'libmorflow_pack_base.so');
      expect(libraryFile('morflow_capi', 'darwin-arm64'), 'libmorflow_capi.dylib');
      expect(libraryFile('morflow_capi', 'windows-x86_64'), 'morflow_capi.dll');
    });
  });

  group('parsePacks', () {
    test('reads pack names and exact versions', () {
      expect(parsePacks(null), isEmpty);
      expect(parsePacks({'base': '0.3.4', 'tensor_stats': '10.20.30'}), {'base': '0.3.4', 'tensor_stats': '10.20.30'});
    });

    test('rejects anything else', () {
      for (final bad in [
        'base',
        ['base'],
        {'base': 'latest'},
        {'base': 0.3},
        {'base': '0.3'},
        {'base': '^0.3.4'},
        {'../base': '0.3.4'},
        {'Base': '0.3.4'},
      ]) {
        expect(() => parsePacks(bad), throwsA(isA<ReleaseException>()), reason: '$bad');
      }
    });
  });

  test('parseChecksums reads sha256sum output in text and binary mode', () {
    final a = sum('a'), b = sum('b');
    expect(parseChecksums('$a  one.so\n$b *two.dll\r\n\nnot a line\n'), {'one.so': a, 'two.dll': b});
  });
}
