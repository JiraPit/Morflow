# Morflow for Unity

Run [Morflow](https://morflow.org) pipelines in your Unity game or app: clean up audio, prepare images and turn data into tensors for your models, offline, on the player's device.

It works in the Editor and in builds for Windows, macOS, Linux, Android and iOS. Your pipelines are ordinary assets, and the engine and the actions they use are bundled into every build, so players need nothing else.

## 1. Add the package

In Unity, open **Window > Package Manager**, choose **+ > Install package from git URL** and enter:

```
https://github.com/JiraPit/Morflow.git?path=bindings/unity#unity-v0.1.0
```

You need Unity 2022.3 or later.

## 2. Add your pipelines

Put your `.morf` files anywhere in `Assets`, for example `Assets/Pipelines/clean.morf`:

```
import audio_basics/latest

accept Audio $audio
$audio >> gain(linear=2.0) >> to_wav >> emit
```

## 3. List the packs they need

Install the [Morflow CLI](https://morflow.org/doc/#cli-installation), then run this in your project's folder:

```
morflow prep --prod-target unity Assets/Pipelines
```

It writes the action packs your pipelines use to `ProjectSettings/Morflow.json`. Commit that file. The next time Unity gets focus, it downloads the engine and those packs into `Assets/Plugins/Morflow`. That folder ignores itself in git, and a fresh clone downloads it again.

## 4. Run a pipeline

Give a script a field for the pipeline, drag the `.morf` file onto it in the Inspector, and run it:

```csharp
using MorflowUnity;
using UnityEngine;

public class Cleaner : MonoBehaviour
{
    public MorflowPipelineAsset clean;

    async void Start()
    {
        using var pipeline = Morflow.FromAsset(clean);

        var clip = GetComponent<AudioSource>().clip;
        var samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);

        var input = TensorInput.Audio(samples, clip.channels, clip.frequency, interleaved: true);
        var wav = await pipeline.RunAsync<MorflowTensor>(input);
        System.IO.File.WriteAllBytes(Application.persistentDataPath + "/clean.wav", wav.ToBytes());
    }
}
```

`RunAsync` works on a background thread, so the game keeps running smoothly, and continues on the main thread when it finishes. `Run` is the same on the current thread.

### Inputs

- A `float[]`, `double[]`, `int[]`, `long[]` (and the other numeric arrays): a one-dimensional tensor of that type.
- A `TensorInput` with a `Shape`, for tensors of any shape, images (`TensorInput.Image`) and audio (`TensorInput.Audio`).
- A `byte[]`: raw bytes, such as a WAV, PNG or JPEG file.
- A `MorflowTensor` from an earlier run.
- An `object[]` of those: a Composite.

### Outputs

`Run` returns a `MorflowTensor`, or a list for a Composite. Read it with `ToFloatArray()`, `ToIntArray()`, `ToBytes()` and so on. Pipelines with several `emit("name")` outputs use `RunToMap`. Parameters declared with `accept` are passed in order with `RunArgs`.

## Platforms

| Platform | Notes |
|---|---|
| Editor | Windows, macOS and Linux |
| Windows | 64-bit |
| macOS | Apple silicon and Intel |
| Linux | 64-bit |
| Android | ARM64 and x86-64, with the IL2CPP scripting backend. Untick ARMv7 under **Player > Other Settings > Target Architectures**. |
| iOS | Devices and the simulator |

WebGL is not supported. Plugin packs (such as `image_opencv`) are not supported yet.

## More

- [Pipeline syntax](https://morflow.org/doc/#syntax)
- [Data types and shapes](https://morflow.org/doc/#types)
- [Report a problem](https://github.com/JiraPit/Morflow/issues)
