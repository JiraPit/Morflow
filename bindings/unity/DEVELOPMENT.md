# Developing the Unity package

The example project in `Example~` uses this package from its folder (`file:../..`), so changes here take effect the next time Unity opens it. The `~` keeps Unity from importing the example as part of the package.

These commands use the [Unity CLI](https://unity.com) (`unity`) and Unity 6000.0.85f2, with the Linux and Android build modules.

## List the example's packs

From `Example~`, with a CLI built from this repository:

```
morflow prep --prod-target unity Assets/Resources/Pipelines
```

## Run the checks in the Editor

```
unity run Example~ -- -executeMethod Builder.RunChecksInEditor
```

The log ends with `MORFLOW DONE PASS`, after one `MORFLOW PASS` line per check in `Assets/Scripts/MorflowChecks.cs`.

## Run the package tests

```
unity test Example~ --mode EditMode
```

## Build and run players

`unity build` does not pass extra arguments to Unity, so builds go through `unity run`:

```
unity run Example~ -- -buildTarget Linux64 -executeMethod Builder.Build -buildOutput Builds/Linux/Example -scriptingBackend IL2CPP
MORFLOW_ACTIONS_PATH=/tmp/empty Example~/Builds/Linux/Example -batchmode -nographics -logFile -
```

An empty `MORFLOW_ACTIONS_PATH` makes sure every action comes from the bundled packs.

For Android:

```
unity run Example~ -- -buildTarget Android -executeMethod Builder.Build -buildOutput Builds/Android/Example.apk
adb install Example~/Builds/Android/Example.apk
adb shell am start -n org.morflow.example/com.unity3d.player.UnityPlayerGameActivity
adb logcat -s Unity:V | grep MORFLOW
```

Run the emulator with a window: on some machines it crashes when started with `-no-window`. Its storage is small, so `adb uninstall org.morflow.example` before installing a new build.

## Libraries from this checkout

To test engine or pack changes before they are released, lay the libraries out like the GitHub releases and point the package at them:

```
python3 scripts/stage_release.py --platform linux-x86_64 --out target/staged-releases
unity run Example~ -- -executeMethod Builder.RunChecksInEditor -morflowReleaseUrl $PWD/target/staged-releases
```

`MORFLOW_RELEASE_URL`, or `"releaseUrl"` in `ProjectSettings/Morflow.json`, does the same.

## CI

`.github/workflows/unity.yml` builds and runs the example on Linux, Android, macOS, iOS and Windows. It needs a Unity license file in the `UNITY_LICENSE` secret, which Unity issues only for Enterprise and Industry seats, so it runs only when started by hand.

## Releasing

Projects install the package from a tag's Git URL. Set the version in `package.json` and `Morflow.PackageVersion`, describe it in `CHANGELOG.md`, then check and tag:

```
python3 scripts/check_unity_release.py --tag unity-v0.1.0
git tag unity-v0.1.0 && git push origin unity-v0.1.0
```
