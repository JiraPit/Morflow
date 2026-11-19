// Converts Dart values to native inputs and native results to Dart values,
// following the same rules as the Node.js binding's index.js.
import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'exception.dart';
import 'tensor.dart';

/// The dtype a typed list carries. A [Uint8List] has none: alone it is raw
/// bytes, and as tensor data it keeps the `'f32'` default.
String? typedDataDtype(Object data) => switch (data) {
  Float32List() => 'f32',
  Float64List() => 'f64',
  Int8List() => 'i8',
  Int16List() => 'i16',
  Int32List() => 'i32',
  Int64List() => 'i64',
  Uint32List() => 'u32',
  Uint64List() => 'u64',
  _ => null,
};

bool _isRawBytes(Object value) => value is Uint8List || value is Uint8ClampedList;

/// A host value normalized to one of the native value kinds.
sealed class _Value {}

final class _Tensor extends _Value {
  final TensorInput input;
  final List<_Tensor> items;
  _Tensor(this.input, [this.items = const []]);
}

final class _Bytes extends _Value {
  final Uint8List bytes;
  _Bytes(this.bytes);
}

final class _Number extends _Value {
  final double value;
  _Number(this.value);
}

final class _Text extends _Value {
  final String value;
  _Text(this.value);
}

final class _Bool extends _Value {
  final bool value;
  _Bool(this.value);
}

const _composite = TensorInput(data: <int>[], shape: [], payloadType: 'composite');

/// A pipeline input: a [TensorInput], typed data (a 1-D tensor of its dtype),
/// a [Uint8List] of raw bytes, or a [List] of those, sent as a Composite.
_Value _input(Object value) {
  if (value is List && !_isRawBytes(value) && value is! TypedData) {
    return _Tensor(_composite, value.map(_component).toList());
  }
  if (_isRawBytes(value)) return _Bytes(value as Uint8List);
  if (value is TypedData) {
    final dtype = typedDataDtype(value);
    if (dtype == null) {
      throw ArgumentError.value(
        value.runtimeType,
        'input',
        'Unsupported typed data; use one of Float32List, Float64List, Int8List, Int16List, Int32List, '
            'Int64List, Uint32List, Uint64List, or Uint8List for raw bytes',
      );
    }
    return _Tensor(TensorInput(data: value, shape: [value.lengthInBytes ~/ value.elementSizeInBytes], dtype: dtype));
  }
  if (value is TensorInput) {
    final data = value.data;
    if (data is! TypedData && data is! ByteBuffer) {
      throw ArgumentError.value(data.runtimeType, 'TensorInput.data', 'Must be typed data or a ByteBuffer');
    }
    return _Tensor(
      TensorInput(
        data: data,
        shape: value.shape,
        dtype: value.dtype ?? typedDataDtype(data),
        payloadType: value.payloadType,
        colorSpace: value.colorSpace,
        sampleRate: value.sampleRate,
        channels: value.channels,
        layout: value.layout,
      ),
    );
  }
  throw ArgumentError.value(
    value.runtimeType,
    'input',
    'Expected a TensorInput, typed data, a Uint8List of raw bytes, or a List (Composite)',
  );
}

_Tensor _component(Object? value) {
  final normalized = value == null ? null : _input(value);
  if (normalized is! _Tensor) {
    throw ArgumentError.value(
      value?.runtimeType,
      'Composite component',
      'Composite components must be TensorInputs, typed data (not raw byte buffers), or Lists',
    );
  }
  return normalized;
}

/// A positional pipeline parameter: anything [_input] accepts, a number
/// (Scalar or `*Arg`), a string (`StrArg`) or a boolean (`BoolArg`).
_Value _arg(Object? value) => switch (value) {
  num() => _Number(value.toDouble()),
  String() => _Text(value),
  bool() => _Bool(value),
  null => throw ArgumentError.notNull('argument'),
  _ => _input(value),
};

Uint8List _bytesOf(Object data) => switch (data) {
  TypedData() => data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
  ByteBuffer() => data.asUint8List(),
  _ => Uint8List(0),
};

Pointer<Utf8> _string(String? text, Allocator arena) => text == null ? nullptr : text.toNativeUtf8(allocator: arena);

Pointer<Uint8> _copy(Uint8List bytes, Allocator arena) {
  if (bytes.isEmpty) return nullptr;
  final pointer = arena<Uint8>(bytes.length);
  pointer.asTypedList(bytes.length).setAll(0, bytes);
  return pointer;
}

void _write(MorflowValueIn target, _Value value, Allocator arena) {
  target
    ..kind = valueTensor
    ..data = nullptr
    ..dataLen = 0
    ..shape = nullptr
    ..rank = 0
    ..dtype = nullptr
    ..payloadType = nullptr
    ..colorSpace = nullptr
    ..layout = nullptr
    ..sampleRate = -1
    ..channels = -1
    ..items = nullptr
    ..itemCount = 0
    ..number = 0
    ..flag = 0;
  switch (value) {
    case _Tensor(:final input, :final items):
      final bytes = _bytesOf(input.data);
      final shape = arena<Uint64>(input.shape.isEmpty ? 1 : input.shape.length);
      for (var i = 0; i < input.shape.length; i++) {
        if (input.shape[i] < 0) throw ArgumentError.value(input.shape, 'shape', 'Dimensions must not be negative');
        shape[i] = input.shape[i];
      }
      target
        ..data = _copy(bytes, arena)
        ..dataLen = bytes.length
        ..shape = shape
        ..rank = input.shape.length
        ..dtype = _string(input.dtype, arena)
        ..payloadType = _string(input.payloadType, arena)
        ..colorSpace = _string(input.colorSpace, arena)
        ..layout = _string(input.layout, arena)
        ..sampleRate = input.sampleRate ?? -1
        ..channels = input.channels ?? -1;
      if (items.isNotEmpty) {
        final list = arena<MorflowValueIn>(items.length);
        for (var i = 0; i < items.length; i++) {
          _write(list[i], items[i], arena);
        }
        target
          ..items = list
          ..itemCount = items.length;
      }
    case _Bytes(:final bytes):
      target
        ..kind = valueBytes
        ..data = _copy(bytes, arena)
        ..dataLen = bytes.length;
    case _Number(:final value):
      target
        ..kind = valueNumber
        ..number = value;
    case _Text(:final value):
      final bytes = utf8.encode(value);
      target
        ..kind = valueText
        ..data = _copy(bytes, arena)
        ..dataLen = bytes.length;
    case _Bool(:final value):
      target
        ..kind = valueBool
        ..flag = value ? 1 : 0;
  }
}

/// The native form of a single-input run's input (null for none).
Pointer<MorflowValueIn> encodeInput(Object? input, Allocator arena) {
  if (input == null) return nullptr;
  final value = arena<MorflowValueIn>();
  _write(value.ref, _input(input), arena);
  return value;
}

/// The native form of positional parameters, in declaration order.
Pointer<MorflowValueIn> encodeArgs(List<Object?> args, Allocator arena) {
  final values = args.map(_arg).toList();
  if (values.isEmpty) return nullptr;
  final list = arena<MorflowValueIn>(values.length);
  for (var i = 0; i < values.length; i++) {
    _write(list[i], values[i], arena);
  }
  return list;
}

Object _readValue(MorflowValueOut value) {
  final dtype = value.dtype.toDartString();
  if (dtype == 'composite') {
    return List<Object>.unmodifiable([for (var i = 0; i < value.itemCount; i++) _readValue(value.items[i])]);
  }
  final shape = [for (var i = 0; i < value.rank; i++) value.shape[i]];
  // Copy out of native memory, which is freed with the result.
  final data = value.dataLen == 0 ? Uint8List(0) : Uint8List.fromList(value.data.asTypedList(value.dataLen));
  return MorflowTensor.fromEngine(shape, dtype, data);
}

/// Reads and frees a result: the single output, or a map of named outputs.
Object decodeResult(Pointer<MorflowResult> result, {required bool all}) {
  try {
    final error = morflowResultError(result);
    if (error != nullptr) throw MorflowException(error.toDartString());
    final count = morflowResultCount(result);
    if (!all) return _readValue(morflowResultValue(result, 0).ref);
    return <String, Object>{
      for (var i = 0; i < count; i++)
        morflowResultName(result, i).toDartString(): _readValue(morflowResultValue(result, i).ref),
    };
  } finally {
    morflowResultFree(result);
  }
}
