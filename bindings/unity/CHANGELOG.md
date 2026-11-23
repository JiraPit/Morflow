# Changelog

## Unreleased

- Runs engine 0.3.6.

## 0.1.0

- First release. Runs Morflow pipelines in the Unity Editor and in Windows, macOS, Linux, Android and iOS builds, with engine 0.3.5.
- `.morf` files in Assets import as `MorflowPipelineAsset`s; `Morflow.FromAsset` compiles them.
- The engine and the packs listed in `ProjectSettings/Morflow.json` (written by `morflow prep --prod-target unity`) are downloaded, checked and bundled into every build.
