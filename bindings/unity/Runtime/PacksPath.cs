// Tells the engine where Unity put the pack libraries, through
// MORFLOW_PACKS_PATH. The engine also looks beside its own library, but Unity
// loads plugins by a path relative to the working directory, which the engine
// does not trust, so on Windows, macOS and Linux the folder is named here.
// Android and iOS need nothing: the engine finds packs by library name and
// in the app's Frameworks folder.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace MorflowUnity
{
    internal static class PacksPath
    {
        private static string dataPath;
        private static bool configured;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void Remember() => dataPath = Application.dataPath;

        /// <summary>Adds the pack folders to MORFLOW_PACKS_PATH before the engine first looks for a pack.</summary>
        internal static void Configure()
        {
            if (configured) return;
            configured = true;
            var folders = new List<string>();
#if UNITY_EDITOR
            // The Editor runs in the project's folder.
            var plugins = Path.Combine(Directory.GetCurrentDirectory(), "Assets", "Plugins", "Morflow");
            switch (Application.platform)
            {
                case RuntimePlatform.LinuxEditor: folders.Add(Path.Combine(plugins, "linux")); break;
                case RuntimePlatform.WindowsEditor: folders.Add(Path.Combine(plugins, "windows")); break;
                case RuntimePlatform.OSXEditor: folders.Add(Path.Combine(plugins, "macos")); break;
            }
#elif UNITY_STANDALONE_OSX
            folders.Add(Path.Combine(Data(), "PlugIns"));
#elif UNITY_STANDALONE
            folders.Add(Path.Combine(Data(), "Plugins", "x86_64"));
            folders.Add(Path.Combine(Data(), "Plugins"));
#endif
            if (folders.Count == 0) return;
            var existing = Environment.GetEnvironmentVariable("MORFLOW_PACKS_PATH");
            if (!string.IsNullOrEmpty(existing)) folders.Add(existing);
            Environment.SetEnvironmentVariable("MORFLOW_PACKS_PATH", string.Join(Path.PathSeparator.ToString(), folders));
        }

        private static string Data() => dataPath ?? Application.dataPath;
    }
}
