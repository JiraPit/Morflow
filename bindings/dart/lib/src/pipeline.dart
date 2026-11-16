import 'dart:async';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'convert.dart';
import 'exception.dart';
import 'version.dart';

final _pipelineFinalizer = NativeFinalizer(
  Native.addressOf<NativeFunction<Void Function(Pointer<MorflowPipelineHandle>)>>(morflowPipelineFree).cast(),
);

bool _versionChecked = false;

/// Fails clearly when the native library and this package disagree, which
/// can only happen when a custom library is configured.
void _checkVersion() {
  if (_versionChecked) return;
  final native = morflowVersion().toDartString();
  if (native != packageVersion) {
    throw MorflowException(
      'The Morflow native library is version $native, but package:morflow is $packageVersion. '
      'Use a native library built from the matching release.',
    );
  }
  _versionChecked = true;
}

Pipeline _compile(Pointer<MorflowPipelineHandle> Function(Pointer<Utf8>, Pointer<Pointer<Utf8>>) compile, String text) {
  _checkVersion();
  return using((arena) {
    final error = arena<Pointer<Utf8>>()..value = nullptr;
    final handle = compile(text.toNativeUtf8(allocator: arena), error);
    if (handle == nullptr) {
      final message = error.value.toDartString();
      morflowStringFree(error.value);
      throw MorflowException(message);
    }
    return Pipeline._(handle);
  });
}

/// Compiles and loads a `.morf` pipeline file from disk.
Pipeline load(String path) => _compile(morflowLoad, path);

/// Compiles a `.morf` pipeline DSL source string directly.
Pipeline fromStr(String source) => _compile(morflowFromStr, source);

/// Alias for [fromStr].
Pipeline fromString(String source) => fromStr(source);

/// Delivers asynchronous run results from native worker threads to this
/// isolate. The callback keeps the isolate alive only while runs are pending.
final class _AsyncRuns {
  static final _AsyncRuns instance = _AsyncRuns();

  final _pending = <int, Completer<Pointer<MorflowResult>>>{};
  int _nextId = 0;
  late final NativeCallable<MorflowCallbackNative> _callback = NativeCallable<MorflowCallbackNative>.listener(_deliver)
    ..keepIsolateAlive = false;

  void _deliver(int requestId, Pointer<MorflowResult> result) {
    final completer = _pending.remove(requestId);
    if (_pending.isEmpty) _callback.keepIsolateAlive = false;
    if (completer == null) {
      morflowResultFree(result);
    } else {
      completer.complete(result);
    }
  }

  /// Starts a run. [start] returns null when the run started, or an error
  /// result when it could not.
  Future<Pointer<MorflowResult>> start(
    Pointer<MorflowResult> Function(Pointer<NativeFunction<MorflowCallbackNative>> callback, int requestId) start,
  ) {
    final requestId = _nextId++;
    final completer = Completer<Pointer<MorflowResult>>();
    _pending[requestId] = completer;
    _callback.keepIsolateAlive = true;
    final immediate = start(_callback.nativeFunction, requestId);
    if (immediate != nullptr) {
      _pending.remove(requestId);
      if (_pending.isEmpty) _callback.keepIsolateAlive = false;
      return Future.value(immediate);
    }
    return completer.future;
  }
}

/// A compiled pipeline.
///
/// Each run method returns the single emitted output: a [MorflowTensor], or a
/// [List] for a Composite. Name the type you expect, for example
/// `pipeline.runSync<MorflowTensor>(input)`. The `…All` methods return every
/// named output.
///
/// A pipeline holds native resources. They are released by [close], or
/// automatically once the pipeline is garbage collected. Runs already in
/// progress finish normally after [close].
final class Pipeline implements Finalizable {
  final Pointer<MorflowPipelineHandle> _handle;
  bool _closed = false;

  /// Parameter names declared with `accept $param`, in declaration order.
  final List<String> params;

  Pipeline._(this._handle) : params = List.unmodifiable(_readParams(_handle)) {
    _pipelineFinalizer.attach(this, _handle.cast(), detach: this);
  }

  static List<String> _readParams(Pointer<MorflowPipelineHandle> handle) {
    return [
      for (var i = 0; i < morflowParamCount(handle); i++)
        () {
          final name = morflowParamName(handle, i);
          try {
            return name.toDartString();
          } finally {
            morflowStringFree(name);
          }
        }(),
    ];
  }

  Pointer<MorflowPipelineHandle> get _live {
    if (_closed) throw StateError('The pipeline has been closed');
    return _handle;
  }

  /// Executes the pipeline synchronously on the current isolate.
  T runSync<T extends Object>([Object? input]) =>
      using((arena) => decodeResult(morflowRun(_live, encodeInput(input, arena), 0), all: false) as T);

  /// Executes the pipeline synchronously, returning all named emitted streams.
  Map<String, T> runSyncAll<T extends Object>([Object? input]) =>
      _named(using((arena) => decodeResult(morflowRun(_live, encodeInput(input, arena), 1), all: true)));

  /// Executes the pipeline on a native worker thread without blocking this
  /// isolate.
  Future<T> run<T extends Object>([Object? input]) async =>
      decodeResult(
            await _start((callback, id, arena) => morflowRunAsync(_live, encodeInput(input, arena), 0, callback, id)),
            all: false,
          )
          as T;

  /// Executes the pipeline on a native worker thread, returning all named
  /// emitted streams.
  Future<Map<String, T>> runAll<T extends Object>([Object? input]) async => _named(
    decodeResult(
      await _start((callback, id, arena) => morflowRunAsync(_live, encodeInput(input, arena), 1, callback, id)),
      all: true,
    ),
  );

  /// Executes the pipeline synchronously with positional parameters bound in
  /// declaration order. Parameters without a supplied value use their
  /// declared default.
  ///
  /// Each argument is anything [runSync] accepts, a number (Scalar or
  /// `*Arg`), a string (`StrArg`) or a boolean (`BoolArg`).
  T runSyncArgs<T extends Object>(List<Object?> args) => using((arena) {
    final values = encodeArgs(args, arena);
    return decodeResult(morflowRunArgs(_live, values, args.length, 0), all: false) as T;
  });

  /// Executes the pipeline synchronously with positional parameters,
  /// returning all named emitted streams.
  Map<String, T> runSyncAllArgs<T extends Object>(List<Object?> args) => _named(
    using((arena) {
      final values = encodeArgs(args, arena);
      return decodeResult(morflowRunArgs(_live, values, args.length, 1), all: true);
    }),
  );

  /// Executes the pipeline on a native worker thread with positional
  /// parameters bound in declaration order.
  Future<T> runArgs<T extends Object>(List<Object?> args) async =>
      decodeResult(
            await _start(
              (callback, id, arena) =>
                  morflowRunArgsAsync(_live, encodeArgs(args, arena), args.length, 0, callback, id),
            ),
            all: false,
          )
          as T;

  /// Executes the pipeline on a native worker thread with positional
  /// parameters, returning all named emitted streams.
  Future<Map<String, T>> runAllArgs<T extends Object>(List<Object?> args) async => _named(
    decodeResult(
      await _start(
        (callback, id, arena) => morflowRunArgsAsync(_live, encodeArgs(args, arena), args.length, 1, callback, id),
      ),
      all: true,
    ),
  );

  /// Releases the pipeline's native resources. Further runs throw a
  /// [StateError]; calling [close] again does nothing.
  void close() {
    if (_closed) return;
    _closed = true;
    _pipelineFinalizer.detach(this);
    morflowPipelineFree(_handle);
  }

  /// Converts inputs (freed as soon as the native call returns, since the
  /// engine copies them) and starts an asynchronous run.
  Future<Pointer<MorflowResult>> _start(
    Pointer<MorflowResult> Function(
      Pointer<NativeFunction<MorflowCallbackNative>> callback,
      int requestId,
      Allocator arena,
    )
    start,
  ) {
    return _AsyncRuns.instance.start((callback, id) => using((arena) => start(callback, id, arena)));
  }

  static Map<String, T> _named<T extends Object>(Object outputs) => Map<String, T>.from(outputs as Map<String, Object>);

  @override
  String toString() => 'Pipeline(params: $params)';
}
