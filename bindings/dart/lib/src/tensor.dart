import 'dart:typed_data';

/// Every tensor element type.
const List<String> tensorDTypes = ['f32', 'f64', 'u8', 'i8', 'i16', 'i32', 'i64', 'u32', 'u64'];

/// A tensor, image or audio input with an explicit shape.
///
/// `data` is typed data or a [ByteBuffer] holding the elements. When `dtype`
/// is omitted it is inferred from typed data other than [Uint8List] (a
/// [Float64List] is `'f64'`, an [Int64List] is `'i64'`, …), and is otherwise
/// `'f32'`.
class TensorInput {
  /// The element bytes: typed data or a [ByteBuffer].
  final Object data;

  /// The dimensions, outermost first.
  final List<int> shape;

  /// Element type of [data]: one of [tensorDTypes].
  final String? dtype;

  /// The payload type to send: `'tensor'`, `'image'` or `'audio'`.
  ///
  /// There is no type inference. A rank-3 array is a plain tensor unless you
  /// ask for `'image'`, and a `[4, N]` float32 feature matrix is a tensor
  /// rather than audio. Defaults to `'tensor'`.
  final String? payloadType;

  /// Color space for `payloadType: 'image'`: `'grayscale'`, `'rgb'`, `'rgba'`,
  /// `'bgr'` or `'bgra'`. Inferred from the channel count when omitted.
  final String? colorSpace;

  /// Sample rate in Hz for `payloadType: 'audio'`. Defaults to 44100.
  final int? sampleRate;

  /// Channel count for `payloadType: 'audio'`. Defaults to the leading
  /// dimension.
  final int? channels;

  /// Memory layout: `'hwc'` / `'chw'` for images, `'planar'` /
  /// `'interleaved'` for audio. Defaults to `'hwc'` and `'planar'`.
  final String? layout;

  const TensorInput({
    required this.data,
    required this.shape,
    this.dtype,
    this.payloadType,
    this.colorSpace,
    this.sampleRate,
    this.channels,
    this.layout,
  });
}

/// An emitted tensor. A Composite output is a [List] of outputs instead.
final class MorflowTensor {
  /// The dimensions, outermost first.
  final List<int> shape;

  /// The element type: one of [tensorDTypes].
  final String dtype;

  /// The raw element bytes, in native byte order.
  final Uint8List data;

  /// Creates a tensor from engine output. Applications receive tensors from
  /// pipeline runs rather than constructing them.
  MorflowTensor.fromEngine(List<int> shape, this.dtype, this.data) : shape = List.unmodifiable(shape);

  /// The number of dimensions.
  int get rank => shape.length;

  /// The number of elements.
  int get length => shape.fold(1, (count, dim) => count * dim);

  /// The elements as the typed list matching [dtype] ([Int64List] for
  /// `'i64'`, [Uint64List] for `'u64'`).
  TypedData toTypedArray() => switch (dtype) {
    'f32' => toFloat32Array(),
    'f64' => toFloat64Array(),
    'u8' => toUint8Array(),
    'i8' => toInt8Array(),
    'i16' => toInt16Array(),
    'i32' => toInt32Array(),
    'i64' => toInt64Array(),
    'u32' => toUint32Array(),
    'u64' => toUint64Array(),
    _ => throw UnsupportedError("No typed list for dtype '$dtype'"),
  };

  Float32List toFloat32Array() => _aligned(4).asFloat32List(0, data.lengthInBytes ~/ 4);
  Float64List toFloat64Array() => _aligned(8).asFloat64List(0, data.lengthInBytes ~/ 8);
  Uint8List toUint8Array() => data;
  Int8List toInt8Array() => data.buffer.asInt8List(data.offsetInBytes, data.lengthInBytes);
  Int16List toInt16Array() => _aligned(2).asInt16List(0, data.lengthInBytes ~/ 2);
  Int32List toInt32Array() => _aligned(4).asInt32List(0, data.lengthInBytes ~/ 4);
  Uint32List toUint32Array() => _aligned(4).asUint32List(0, data.lengthInBytes ~/ 4);
  Int64List toInt64Array() => _aligned(8).asInt64List(0, data.lengthInBytes ~/ 8);
  Uint64List toUint64Array() => _aligned(8).asUint64List(0, data.lengthInBytes ~/ 8);

  /// The raw element bytes.
  Uint8List toBuffer() => data;

  /// A buffer starting at [data]'s first byte, copied only when [data] is
  /// not aligned to the element size.
  ByteBuffer _aligned(int elementSize) {
    if (data.offsetInBytes == 0) return data.buffer;
    return Uint8List.fromList(data).buffer;
  }

  @override
  String toString() => 'MorflowTensor(dtype: $dtype, shape: $shape)';
}
