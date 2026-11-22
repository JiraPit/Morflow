// When the Morflow libraries are prepared:
//   - when the project opens and when the Editor gets focus again (after
//     `morflow prep --prod-target unity` changed ProjectSettings/Morflow.json),
//     for the Editor's own platform and the active build target;
//   - when the build target changes;
//   - before every build, which also rejects settings Morflow cannot run with.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace MorflowUnity.Editor
{
    /// <summary>Editor functions for build scripts and CI.</summary>
    public static class MorflowEditor
    {
        /// <summary>
        /// Downloads and bundles the engine and the packs in
        /// ProjectSettings/Morflow.json for the Editor and the active build
        /// target now, instead of waiting for the Editor to do it. Throws when
        /// that fails. Builds do this themselves.
        /// </summary>
        public static void RefreshNativeLibraries() =>
            NativeLibraries.Sync(Hooks.CurrentGroups(), Settings.Read(Hooks.ProjectRoot), Hooks.ProjectRoot);
    }

    [InitializeOnLoad]
    internal static class Hooks
    {
        private static string lastSettings;

        static Hooks()
        {
            EditorApplication.delayCall += () => Refresh(false);
            EditorApplication.focusChanged += focused =>
            {
                if (focused) Refresh(false);
            };
            EditorApplication.playModeStateChanged += change =>
            {
                if (change == PlayModeStateChange.ExitingEditMode) Refresh(false);
            };
        }

        internal static string ProjectRoot => Path.GetDirectoryName(Application.dataPath);

        /// <summary>The groups the Editor needs now: its own platform and the active build target.</summary>
        internal static IEnumerable<LibraryGroup> CurrentGroups()
        {
            var groups = new List<LibraryGroup>();
            if (NativeLibraries.EditorGroup() is LibraryGroup editor) groups.Add(editor);
            if (NativeLibraries.TargetGroup(EditorUserBuildSettings.activeBuildTarget) is LibraryGroup target) groups.Add(target);
            return groups;
        }

        [MenuItem("Tools/Morflow/Refresh Native Libraries")]
        private static void RefreshFromMenu() => Refresh(true);

        /// <summary>Brings the libraries up to date, logging problems instead of throwing.</summary>
        internal static void Refresh(bool force)
        {
            // Libraries cannot change while the game runs in the Editor.
            if (EditorApplication.isPlaying && !force) return;
            try
            {
                var settingsPath = Path.Combine(ProjectRoot, Settings.File);
                var stamp = File.Exists(settingsPath) ? File.ReadAllText(settingsPath) : "";
                stamp += "|" + EditorUserBuildSettings.activeBuildTarget + "|" + PlayerSettings.iOS.sdkVersion;
                if (!force && stamp == lastSettings) return;
                NativeLibraries.Sync(CurrentGroups(), Settings.Read(ProjectRoot), ProjectRoot);
                lastSettings = stamp;
            }
            catch (Exception e) when (e is ReleaseException || e is SettingsException || e is IOException)
            {
                Debug.LogError($"Morflow: {e.Message}");
            }
        }

        internal sealed class TargetChanged : IActiveBuildTargetChanged
        {
            public int callbackOrder => 0;

            public void OnActiveBuildTargetChanged(BuildTarget previousTarget, BuildTarget newTarget) => Refresh(false);
        }

        internal sealed class BeforeBuild : IPreprocessBuildWithReport
        {
            public int callbackOrder => -100;

            public void OnPreprocessBuild(BuildReport report)
            {
                var target = report.summary.platform;
                if (!(NativeLibraries.TargetGroup(target) is LibraryGroup group))
                {
                    throw new BuildFailedException(
                        $"Morflow runs on Windows (64-bit), macOS, Linux (64-bit), Android and iOS, not {target}. Remove the Morflow package to build for {target}.");
                }
                Check(target);
                try
                {
                    // The build needs only its target's libraries.
                    NativeLibraries.Sync(new[] { group }, Settings.Read(ProjectRoot), ProjectRoot);
                }
                catch (Exception e) when (e is ReleaseException || e is SettingsException || e is IOException)
                {
                    throw new BuildFailedException($"Morflow: {e.Message}");
                }
            }

            private static void Check(BuildTarget target)
            {
                if (target != BuildTarget.Android) return;
                var architectures = PlayerSettings.Android.targetArchitectures;
                var unsupported = new List<string>();
                if ((architectures & AndroidArchitecture.ARMv7) != 0) unsupported.Add("ARMv7");
#if UNITY_2023_1_OR_NEWER
                if ((architectures & AndroidArchitecture.X86) != 0) unsupported.Add("x86");
#endif
                if (unsupported.Count > 0)
                {
                    throw new BuildFailedException(
                        $"Morflow's Android libraries are for ARM64 and x86-64 only. In Project Settings > Player > Android > Other Settings > Target Architectures, untick {string.Join(" and ", unsupported)} (ARM64 needs the IL2CPP scripting backend).");
                }
                if ((architectures & (AndroidArchitecture.ARM64 | X86_64())) == 0)
                {
                    throw new BuildFailedException("Tick ARM64 in Project Settings > Player > Android > Other Settings > Target Architectures (it needs the IL2CPP scripting backend).");
                }
            }

            private static AndroidArchitecture X86_64()
            {
#if UNITY_2023_1_OR_NEWER
                return AndroidArchitecture.X86_64;
#else
                return Enum.TryParse<AndroidArchitecture>("X86_64", out var value) ? value : 0;
#endif
            }
        }
    }
}
