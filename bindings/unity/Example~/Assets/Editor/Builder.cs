// Command-line entry points for CI and local testing, for example:
//   unity run . -- -executeMethod Builder.RunChecksInEditor
//   unity build . --target StandaloneLinux64 --execute-method Builder.Build -o Builds/Linux/Example
// Build takes the target from -buildTarget and the output from -buildOutput.
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class Builder
{
    private const string Scene = "Assets/Main.unity";

    /// <summary>Runs the example's checks in the Editor and exits with their result.</summary>
    public static void RunChecksInEditor()
    {
        MorflowUnity.Editor.MorflowEditor.RefreshNativeLibraries();
        // Batch mode quits when this method returns, so wait here, with the
        // checks continuing on worker threads instead of the blocked main thread.
        var context = System.Threading.SynchronizationContext.Current;
        System.Threading.SynchronizationContext.SetSynchronizationContext(null);
        bool ok;
        try
        {
            ok = MorflowChecks.RunAll().GetAwaiter().GetResult();
        }
        finally
        {
            System.Threading.SynchronizationContext.SetSynchronizationContext(context);
        }
        EditorApplication.Exit(ok ? 0 : 1);
    }

    public static void Build()
    {
        var target = EditorUserBuildSettings.activeBuildTarget;
        var output = Argument("-buildOutput") ?? $"Builds/{target}/Example";
        if (!File.Exists(Scene))
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            EditorSceneManager.SaveScene(scene, Scene);
        }
        PlayerSettings.productName = "MorflowExample";
        PlayerSettings.SetApplicationIdentifier(BuildPipeline.GetBuildTargetGroup(target), "org.morflow.example");
        if (target == BuildTarget.Android)
        {
            PlayerSettings.SetScriptingBackend(UnityEditor.Build.NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64 | AndroidArchitecture.X86_64;
            // Emulators on CI runners render without a GPU; Vulkan stalls there.
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { UnityEngine.Rendering.GraphicsDeviceType.OpenGLES3 });
            if (!output.EndsWith(".apk")) output += ".apk";
        }
        if (target == BuildTarget.iOS)
        {
            PlayerSettings.iOS.targetOSVersionString = "13.0";
            if (Array.IndexOf(Environment.GetCommandLineArgs(), "-iosSimulator") >= 0)
            {
                PlayerSettings.iOS.sdkVersion = iOSSdkVersion.SimulatorSDK;
            }
        }
        var backend = Argument("-scriptingBackend");
        if (backend != null)
        {
            var named = UnityEditor.Build.NamedBuildTarget.FromBuildTargetGroup(BuildPipeline.GetBuildTargetGroup(target));
            PlayerSettings.SetScriptingBackend(named, backend == "IL2CPP" ? ScriptingImplementation.IL2CPP : ScriptingImplementation.Mono2x);
        }
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { Scene },
            locationPathName = output,
            target = target,
            options = BuildOptions.None,
        });
        Debug.Log($"Build {report.summary.result}: {output}");
        EditorApplication.Exit(report.summary.result == BuildResult.Succeeded ? 0 : 1);
    }

    private static string Argument(string name)
    {
        var args = Environment.GetCommandLineArgs();
        var at = Array.IndexOf(args, name);
        return at >= 0 && at + 1 < args.Length ? args[at + 1] : null;
    }
}
