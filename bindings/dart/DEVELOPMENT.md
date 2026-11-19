# Developing the Dart package

These notes are for work inside the Morflow repository; they are not published to pub.dev.

Inside a Morflow checkout, the build hook compiles the native library with `cargo`. Tests that execute actions need an action cache built from the checkout:

```sh
bash scripts/build-actions.sh
python3 scripts/prepare_test_actions.py --profile release --cache /tmp/morflow-actions
cd bindings/dart && MORFLOW_ACTIONS_PATH=/tmp/morflow-actions dart test
```

The Flutter app in `example/` runs every bundleable pack on a device. To test it against libraries built from the checkout, stage them like releases and add `release_url: ../../../target/staged-releases` to its `hooks.user_defines.morflow`:

```sh
python3 scripts/stage_release.py --platform android-x86_64 --out target/staged-releases
cd bindings/dart/example && flutter test integration_test
```
