// Runs the pipelines in Assets/Resources/Pipelines with the engine and packs
// bundled into the game, and logs a "MORFLOW" line per check, which CI reads
// from the Player log (desktop), logcat (Android) or the simulator log (iOS).
// Desktop builds quit when the checks finish.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MorflowUnity;
using UnityEngine;

public static class MorflowChecks
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static async void OnGameStart()
    {
        var ok = await RunAll();
#if !UNITY_EDITOR && (UNITY_STANDALONE)
        Application.Quit(ok ? 0 : 1);
#endif
    }

    /// <summary>Runs every check and logs the results. True when all pass.</summary>
    public static async Task<bool> RunAll()
    {
        // Loaded up front, on the main thread, as Unity requires.
        var assets = new Dictionary<string, MorflowPipelineAsset>();
        foreach (var name in new[] { "abs", "boost", "add", "relu", "identity" })
        {
            assets[name] = Resources.Load<MorflowPipelineAsset>($"Pipelines/{name}");
        }
        MorflowPipelineAsset Pipeline(string name) => assets[name] ?? throw new Exception($"Pipelines/{name}.morf was not imported");

        var results = new List<(string name, bool ok, string detail)>();
        async Task Check(string name, Func<Task<string>> body)
        {
            try
            {
                results.Add((name, true, await body()));
            }
            catch (Exception e)
            {
                results.Add((name, false, e.Message));
            }
        }

        Debug.Log($"MORFLOW engine {Morflow.EngineVersion} on {Application.platform}");

        await Check("abs on a worker thread", async () =>
        {
            using var pipeline = Morflow.FromAsset(Pipeline("abs"));
            var values = (await pipeline.RunAsync<MorflowTensor>(new[] { -1.5f, 2f, -3f })).ToFloatArray();
            if (string.Join(" ", values) != "1.5 2 3") throw new Exception($"got {string.Join(" ", values)}");
            return $"abs: {string.Join(" ", values)}";
        });

        await Check("abs on the main thread", () =>
        {
            using var pipeline = Morflow.FromAsset(Pipeline("abs"));
            var output = pipeline.Run<MorflowTensor>(new TensorInput(new double[] { -4, 5 }, 2));
            var values = output.ToDoubleArray();
            if (output.DType != "f64" || string.Join(" ", values) != "4 5") throw new Exception($"got {output.DType} {string.Join(" ", values)}");
            return Task.FromResult($"{output.DType}: {string.Join(" ", values)}");
        });

        await Check("audio gain to WAV", async () =>
        {
            using var pipeline = Morflow.FromAsset(Pipeline("boost"));
            var samples = Enumerable.Range(0, 2205).Select(i => 0.25f * Mathf.Sin(i / 10f)).ToArray();
            var wav = (await pipeline.RunAsync<MorflowTensor>(TensorInput.Audio(samples, 1, 22050))).ToBytes();
            var header = Encoding.ASCII.GetString(wav, 0, 4);
            if (header != "RIFF") throw new Exception($"not a WAV file ({header})");
            return $"{wav.Length}-byte WAV";
        });

        await Check("Composite add", async () =>
        {
            using var pipeline = Morflow.FromAsset(Pipeline("add"));
            var output = await pipeline.RunAsync<MorflowTensor>(new object[] { new TensorInput(new[] { 1, 2, 3, 4, 5, 6 }, 2, 3), new[] { 10, 20, 30 } });
            var values = string.Join(" ", output.ToIntArray());
            if (values != "11 22 33 14 25 36") throw new Exception($"got {values}");
            return values;
        });

        await Check("scalar argument", async () =>
        {
            using var pipeline = Morflow.FromAsset(Pipeline("relu"));
            var output = (MorflowTensor)await pipeline.RunArgsAsync(-3.0);
            if (output.ToFloatArray()[0] != 0f) throw new Exception($"got {output.ToFloatArray()[0]}");
            return "relu(-3) = 0";
        });

        await Check("exact bundled version", () =>
        {
            using var pipeline = Morflow.FromAsset(Pipeline("identity"));
            var value = pipeline.Run<MorflowTensor>(new[] { 4f }).ToFloatArray()[0];
            if (value != 4f) throw new Exception($"got {value}");
            return Task.FromResult("base identity");
        });

        await Check("errors reach C#", () =>
        {
            try
            {
                Morflow.FromStr("accept Tensor $t\n$t >> no_such_action >> emit");
            }
            catch (MorflowException e)
            {
                return Task.FromResult(e.Message.Length > 60 ? e.Message.Substring(0, 60) : e.Message);
            }
            throw new Exception("compiling an unknown action did not fail");
        });

        foreach (var (name, ok, detail) in results) Debug.Log($"MORFLOW {(ok ? "PASS" : "FAIL")} {name}: {detail}");
        var all = results.All(r => r.ok);
        Debug.Log($"MORFLOW DONE {(all ? "PASS" : "FAIL")}");
        return all;
    }
}
