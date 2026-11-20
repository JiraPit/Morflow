## 0.3.4

- Runs engine 0.3.4. The Morflow CLI 0.3.4 can manage the packs your app ships: `morflow list`, `install`, `uninstall` and `clean` with `--prod-target flutter`.

## 0.3.3+2

- A shorter README: a step-by-step start, with links to the documentation.
- Runs engine 0.3.3.

## 0.3.3+1

- Install instructions for Flutter apps (`flutter pub add morflow`).
- Runs engine 0.3.3, like 0.3.3.

## 0.3.3

- First Dart release: `load`, `fromStr`/`fromString` and `Pipeline` with the same
  run methods and input rules as the Node.js package.
- Runs on Linux, macOS and Windows, and in Flutter apps on Android, iOS, Linux,
  macOS and Windows.
- Apps bundle the action packs listed under `hooks.user_defines.morflow.packs`;
  `morflow prep --prod-target flutter` writes that list.
