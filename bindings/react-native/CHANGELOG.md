## Unreleased

- `.morf` imports type-check in TypeScript apps again. The 0.3.5 typings lost the line that declares `*.morf` modules, so `import boost from './boost.morf'` failed with TS2307.

## 0.3.5

- `runAll`, `runAllArgs`, `runSyncAll` and `runSyncAllArgs` are renamed to `runToMap`, `runToMapArgs`, `runSyncToMap` and `runSyncToMapArgs`, which say what they return: every named output, keyed by name. The old names still work but are deprecated.
- Runs engine 0.3.5.

## 0.3.4

- First release: run Morflow pipelines in React Native apps on Android and iOS, with the same API as the Node.js package. Runs engine 0.3.4.
