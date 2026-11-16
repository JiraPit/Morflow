/// Morflow: data processing and tensor transformation pipelines.
///
/// Compile a `.morf` pipeline with [load] or [fromStr], then run it with the
/// methods on [Pipeline]:
///
/// ```dart
/// import 'dart:typed_data';
/// import 'package:morflow/morflow.dart' as morflow;
///
/// final pipeline = morflow.fromStr('''
///   accept Tensor $tensor
///   $tensor >> base/latest/identity >> emit
/// ''');
/// final output = await pipeline.run<morflow.MorflowTensor>(Float32List.fromList([1, 2, 3]));
/// print(output.toFloat32Array());
/// ```
///
/// Run inputs are a [TensorInput], typed data (a 1-D tensor of its dtype), a
/// `Uint8List` of raw bytes, or a `List` of those, sent as a Composite.
library;

export 'src/exception.dart' show MorflowException;
export 'src/pipeline.dart' show Pipeline, load, fromStr, fromString;
export 'src/tensor.dart' show MorflowTensor, TensorInput, tensorDTypes;
