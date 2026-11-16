// Declarations of the Morflow C interface (bindings/capi). Internal to the
// package: applications use the API in package:morflow/morflow.dart.
//
// The native library is provided by hook/build.dart under this file's asset
// id. Struct layouts must match the #[repr(C)] types in bindings/capi.
@DefaultAsset('package:morflow/src/bindings.dart')
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

/// A tensor, image, audio or Composite input (`MORFLOW_VALUE_TENSOR`).
const int valueTensor = 0;

/// Raw bytes (`MORFLOW_VALUE_BYTES`).
const int valueBytes = 1;

/// A number bound to a pipeline parameter (`MORFLOW_VALUE_NUMBER`).
const int valueNumber = 2;

/// A string bound to a pipeline parameter (`MORFLOW_VALUE_TEXT`).
const int valueText = 3;

/// A boolean bound to a pipeline parameter (`MORFLOW_VALUE_BOOL`).
const int valueBool = 4;

final class MorflowValueIn extends Struct {
  @Uint32()
  external int kind;
  external Pointer<Uint8> data;
  @Size()
  external int dataLen;
  external Pointer<Uint64> shape;
  @Size()
  external int rank;
  external Pointer<Utf8> dtype;
  external Pointer<Utf8> payloadType;
  external Pointer<Utf8> colorSpace;
  external Pointer<Utf8> layout;
  @Int64()
  external int sampleRate;
  @Int64()
  external int channels;
  external Pointer<MorflowValueIn> items;
  @Size()
  external int itemCount;
  @Double()
  external double number;
  @Uint8()
  external int flag;
}

final class MorflowValueOut extends Struct {
  external Pointer<Utf8> dtype;
  external Pointer<Uint64> shape;
  @Size()
  external int rank;
  external Pointer<Uint8> data;
  @Size()
  external int dataLen;
  external Pointer<MorflowValueOut> items;
  @Size()
  external int itemCount;
}

final class MorflowPipelineHandle extends Opaque {}

final class MorflowResult extends Opaque {}

typedef MorflowCallbackNative = Void Function(Int64, Pointer<MorflowResult>);

@Native<Pointer<Utf8> Function()>(symbol: 'morflow_version', isLeaf: true)
external Pointer<Utf8> morflowVersion();

@Native<Pointer<MorflowPipelineHandle> Function(Pointer<Utf8>, Pointer<Pointer<Utf8>>)>(symbol: 'morflow_load')
external Pointer<MorflowPipelineHandle> morflowLoad(Pointer<Utf8> path, Pointer<Pointer<Utf8>> errorOut);

@Native<Pointer<MorflowPipelineHandle> Function(Pointer<Utf8>, Pointer<Pointer<Utf8>>)>(symbol: 'morflow_from_str')
external Pointer<MorflowPipelineHandle> morflowFromStr(Pointer<Utf8> source, Pointer<Pointer<Utf8>> errorOut);

@Native<Void Function(Pointer<MorflowPipelineHandle>)>(symbol: 'morflow_pipeline_free')
external void morflowPipelineFree(Pointer<MorflowPipelineHandle> pipeline);

@Native<Size Function(Pointer<MorflowPipelineHandle>)>(symbol: 'morflow_param_count', isLeaf: true)
external int morflowParamCount(Pointer<MorflowPipelineHandle> pipeline);

@Native<Pointer<Utf8> Function(Pointer<MorflowPipelineHandle>, Size)>(symbol: 'morflow_param_name', isLeaf: true)
external Pointer<Utf8> morflowParamName(Pointer<MorflowPipelineHandle> pipeline, int index);

@Native<Void Function(Pointer<Utf8>)>(symbol: 'morflow_string_free', isLeaf: true)
external void morflowStringFree(Pointer<Utf8> text);

@Native<Pointer<MorflowResult> Function(Pointer<MorflowPipelineHandle>, Pointer<MorflowValueIn>, Uint8)>(
  symbol: 'morflow_run',
)
external Pointer<MorflowResult> morflowRun(
  Pointer<MorflowPipelineHandle> pipeline,
  Pointer<MorflowValueIn> input,
  int all,
);

@Native<Pointer<MorflowResult> Function(Pointer<MorflowPipelineHandle>, Pointer<MorflowValueIn>, Size, Uint8)>(
  symbol: 'morflow_run_args',
)
external Pointer<MorflowResult> morflowRunArgs(
  Pointer<MorflowPipelineHandle> pipeline,
  Pointer<MorflowValueIn> args,
  int count,
  int all,
);

@Native<
  Pointer<MorflowResult> Function(
    Pointer<MorflowPipelineHandle>,
    Pointer<MorflowValueIn>,
    Uint8,
    Pointer<NativeFunction<MorflowCallbackNative>>,
    Int64,
  )
>(symbol: 'morflow_run_async')
external Pointer<MorflowResult> morflowRunAsync(
  Pointer<MorflowPipelineHandle> pipeline,
  Pointer<MorflowValueIn> input,
  int all,
  Pointer<NativeFunction<MorflowCallbackNative>> callback,
  int requestId,
);

@Native<
  Pointer<MorflowResult> Function(
    Pointer<MorflowPipelineHandle>,
    Pointer<MorflowValueIn>,
    Size,
    Uint8,
    Pointer<NativeFunction<MorflowCallbackNative>>,
    Int64,
  )
>(symbol: 'morflow_run_args_async')
external Pointer<MorflowResult> morflowRunArgsAsync(
  Pointer<MorflowPipelineHandle> pipeline,
  Pointer<MorflowValueIn> args,
  int count,
  int all,
  Pointer<NativeFunction<MorflowCallbackNative>> callback,
  int requestId,
);

@Native<Pointer<Utf8> Function(Pointer<MorflowResult>)>(symbol: 'morflow_result_error', isLeaf: true)
external Pointer<Utf8> morflowResultError(Pointer<MorflowResult> result);

@Native<Size Function(Pointer<MorflowResult>)>(symbol: 'morflow_result_count', isLeaf: true)
external int morflowResultCount(Pointer<MorflowResult> result);

@Native<Pointer<Utf8> Function(Pointer<MorflowResult>, Size)>(symbol: 'morflow_result_name', isLeaf: true)
external Pointer<Utf8> morflowResultName(Pointer<MorflowResult> result, int index);

@Native<Pointer<MorflowValueOut> Function(Pointer<MorflowResult>, Size)>(symbol: 'morflow_result_value', isLeaf: true)
external Pointer<MorflowValueOut> morflowResultValue(Pointer<MorflowResult> result, int index);

@Native<Void Function(Pointer<MorflowResult>)>(symbol: 'morflow_result_free')
external void morflowResultFree(Pointer<MorflowResult> result);
