# react-native-morflow example

An app that runs the pipelines in `pipelines/` and shows the results. CI runs
it on an Android emulator and the iOS simulator and checks the `MORFLOW` lines
it logs.

```sh
# From bindings/react-native: pack the package the app installs.
npm pack && mv react-native-morflow-*.tgz react-native-morflow.tgz
cd example && npm install
npx react-native run-android   # or: cd ios && pod install && cd .. && npx react-native run-ios
```

The packs the pipelines need are listed in `package.json` under
`morflow.packs`, written by `morflow prep --prod-target react-native pipelines/*.morf`.
