/// The engine version this package runs: its build hook downloads the native
/// library from the engine release `v<engineVersion>`, and the library must
/// report this version. Bump it with the engine.
///
/// The package's own version in pubspec.yaml is the engine version, or the
/// engine version plus a build number (0.3.3+1) for a release that changes
/// only the Dart package.
const String engineVersion = '0.3.4';
