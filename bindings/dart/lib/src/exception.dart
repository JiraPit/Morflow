/// An error reported by the Morflow engine: a pipeline that fails to parse or
/// compile, an action error, or an input it cannot accept.
class MorflowException implements Exception {
  /// The engine's message, such as `Parse error: …` or `Unknown dtype 'f16' …`.
  final String message;

  const MorflowException(this.message);

  @override
  String toString() => 'MorflowException: $message';
}
