// Puts the Morflow engine and the packs listed in ProjectSettings/Morflow.json
// into Assets/Plugins/Morflow, one folder per platform, and sets each
// library's plugin import settings so Unity loads it in the Editor and ships
// it in builds. The engine finds pack libraries where Unity puts them: beside
// itself on Windows, macOS and Linux, among the app's libraries on Android,
// and as Frameworks/morflow_pack_<pack>.framework on iOS.
//
// The folder is generated: its .gitignore keeps the libraries out of version
// control, and a fresh clone downloads them again when the project opens.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace MorflowUnity.Editor
{
    /// <summary>A set of platforms whose libraries share one folder and one way of bundling.</summary>
    internal enum LibraryGroup
    {
        Linux,
        Windows,
        MacOS,
        Android,
        IOS,
    }

    internal static class NativeLibraries
    {
        internal const string Root = "Assets/Plugins/Morflow";
        private const string StateFile = ".state";
        // Bump when the layout or import settings change, so folders are rebuilt.
        private const int LayoutVersion = 1;

        /// <summary>One library to bundle: the engine or a pack.</summary>
        private sealed class Library
        {
            public string Stem;
            public string Version;
            public Func<string, string> Tag;
            public Func<string, string> Asset;
        }

        /// <summary>The group the Editor itself runs on, or null for an Editor Morflow has no libraries for.</summary>
        internal static LibraryGroup? EditorGroup()
        {
            if (SystemInfo.processorType.IndexOf("ARM", StringComparison.OrdinalIgnoreCase) >= 0 && Application.platform != RuntimePlatform.OSXEditor)
            {
                return null;
            }
            switch (Application.platform)
            {
                case RuntimePlatform.LinuxEditor: return LibraryGroup.Linux;
                case RuntimePlatform.WindowsEditor: return LibraryGroup.Windows;
                case RuntimePlatform.OSXEditor: return LibraryGroup.MacOS;
                default: return null;
            }
        }

        /// <summary>The group a build target needs, or null when Morflow does not run there.</summary>
        internal static LibraryGroup? TargetGroup(BuildTarget target)
        {
            switch (target)
            {
                case BuildTarget.StandaloneLinux64: return LibraryGroup.Linux;
                case BuildTarget.StandaloneWindows64: return LibraryGroup.Windows;
                case BuildTarget.StandaloneOSX: return LibraryGroup.MacOS;
                case BuildTarget.Android: return LibraryGroup.Android;
                case BuildTarget.iOS: return LibraryGroup.IOS;
                default: return null;
            }
        }

        internal static string Folder(LibraryGroup group) => $"{Root}/{group.ToString().ToLowerInvariant()}";

        /// <summary>
        /// Makes each group's folder hold exactly the engine and the listed
        /// packs, downloading what is missing. Folders already up to date are
        /// left alone, so this is cheap to call often.
        /// </summary>
        /// <returns>True when any library changed.</returns>
        internal static bool Sync(IEnumerable<LibraryGroup> groups, Settings settings, string projectRoot)
        {
            var libraries = new List<Library>
            {
                new Library
                {
                    Stem = "morflow_capi",
                    Version = Morflow.EngineVersion,
                    Tag = _ => Releases.EngineTag(Morflow.EngineVersion),
                    Asset = Releases.CapiAsset,
                },
            };
            foreach (var pack in settings.Packs)
            {
                libraries.Add(new Library
                {
                    Stem = $"morflow_pack_{pack.Key}",
                    Version = pack.Value,
                    Tag = _ => Releases.PackTag(pack.Key, pack.Value),
                    Asset = platform => Releases.PackAsset(pack.Key, pack.Value, platform),
                });
            }

            var changed = new List<LibraryGroup>();
            using (var releases = new Releases(settings.ReleaseUrl, Path.Combine(projectRoot, "Library", "Morflow", "releases")))
            {
                foreach (var group in groups.Distinct())
                {
                    var folder = Path.Combine(projectRoot, Folder(group));
                    var state = State(group, libraries);
                    var statePath = Path.Combine(folder, StateFile);
                    if (File.Exists(statePath) && File.ReadAllText(statePath) == state && Expected(group, libraries).All(f => Exists(Path.Combine(folder, f))))
                    {
                        continue;
                    }
                    try
                    {
                        EditorUtility.DisplayProgressBar("Morflow", $"Preparing the {group} libraries", 0.5f);
                        Write(group, libraries, folder, releases);
                    }
                    finally
                    {
                        EditorUtility.ClearProgressBar();
                    }
                    File.WriteAllText(statePath, state);
                    changed.Add(group);
                }
            }
            if (changed.Count == 0) return false;

            WriteGitignore(Path.Combine(projectRoot, Root));
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            foreach (var group in changed)
            {
                foreach (var file in Expected(group, libraries)) Configure(group, $"{Folder(group)}/{file}");
            }
            var packs = settings.Packs.Count == 0 ? "none" : string.Join(", ", settings.Packs.Select(p => $"{p.Key} {p.Value}"));
            Debug.Log($"Morflow: prepared engine {Morflow.EngineVersion} and packs ({packs}) for {string.Join(", ", changed)}.");
            return true;
        }

        /// <summary>What a folder's libraries were made from; a change rebuilds the folder.</summary>
        private static string State(LibraryGroup group, List<Library> libraries)
        {
            var parts = libraries.Select(l => $"{l.Stem}={l.Version}").ToList();
            parts.Insert(0, $"layout {LayoutVersion}");
            if (group == LibraryGroup.IOS) parts.Add($"slice {IosSlice()}");
            return string.Join("\n", parts) + "\n";
        }

        /// <summary>The files (or iOS framework folders) a group's folder holds, relative to it.</summary>
        private static IEnumerable<string> Expected(LibraryGroup group, List<Library> libraries)
        {
            foreach (var library in libraries)
            {
                switch (group)
                {
                    case LibraryGroup.Linux:
                        yield return Releases.LibraryFile(library.Stem, "linux-x86_64");
                        break;
                    case LibraryGroup.Windows:
                        yield return Releases.LibraryFile(library.Stem, "windows-x86_64");
                        break;
                    case LibraryGroup.MacOS:
                        yield return Releases.LibraryFile(library.Stem, "darwin-arm64");
                        break;
                    case LibraryGroup.Android:
                        yield return $"arm64-v8a/{Releases.LibraryFile(library.Stem, "android-aarch64")}";
                        yield return $"x86_64/{Releases.LibraryFile(library.Stem, "android-x86_64")}";
                        break;
                    case LibraryGroup.IOS:
                        yield return $"{library.Stem}.framework";
                        break;
                }
            }
        }

        private static bool Exists(string path) => File.Exists(path) || Directory.Exists(path);

        private static string IosSlice() =>
            PlayerSettings.iOS.sdkVersion == iOSSdkVersion.SimulatorSDK ? "ios-arm64-sim" : "ios-arm64";

        private static void Write(LibraryGroup group, List<Library> libraries, string folder, Releases releases)
        {
            // Fetch everything first, so a failed download leaves the folder as it was.
            var files = new Dictionary<string, byte[]>();
            foreach (var library in libraries)
            {
                byte[] Get(string platform) => File.ReadAllBytes(releases.Fetch(library.Tag(platform), library.Asset(platform)));
                switch (group)
                {
                    case LibraryGroup.Linux:
                        files[Releases.LibraryFile(library.Stem, "linux-x86_64")] = Get("linux-x86_64");
                        break;
                    case LibraryGroup.Windows:
                        files[Releases.LibraryFile(library.Stem, "windows-x86_64")] = Get("windows-x86_64");
                        break;
                    case LibraryGroup.MacOS:
                        // Universal, so the Editor and standalone builds work on Apple silicon and Intel Macs.
                        files[Releases.LibraryFile(library.Stem, "darwin-arm64")] = MachO.Universal(new[] { Get("darwin-arm64"), Get("darwin-x86_64") });
                        break;
                    case LibraryGroup.Android:
                        files[$"arm64-v8a/{Releases.LibraryFile(library.Stem, "android-aarch64")}"] = Get("android-aarch64");
                        files[$"x86_64/{Releases.LibraryFile(library.Stem, "android-x86_64")}"] = Get("android-x86_64");
                        break;
                    case LibraryGroup.IOS:
                        var slice = IosSlice();
                        var binary = Get(slice);
                        // The name the app links and embeds the framework by.
                        MachO.SetInstallName(binary, $"@rpath/{library.Stem}.framework/{library.Stem}");
                        files[$"{library.Stem}.framework/{library.Stem}"] = binary;
                        files[$"{library.Stem}.framework/Info.plist"] = System.Text.Encoding.UTF8.GetBytes(
                            InfoPlist(library.Stem, library.Version, slice == "ios-arm64-sim" ? "iPhoneSimulator" : "iPhoneOS"));
                        break;
                }
            }

            Directory.CreateDirectory(folder);
            var keep = new HashSet<string>(files.Keys.Select(Normalize));
            // Remove libraries of packs no longer listed (with their .meta files).
            foreach (var path in Directory.GetFiles(folder, "*", SearchOption.AllDirectories))
            {
                var relative = Normalize(path.Substring(folder.Length + 1));
                var owner = relative.EndsWith(".meta", StringComparison.Ordinal) ? relative.Substring(0, relative.Length - 5) : relative;
                if (relative == StateFile || keep.Contains(owner) || keep.Any(k => k.StartsWith(owner + "/", StringComparison.Ordinal))) continue;
                Delete(path);
            }
            foreach (var dir in Directory.GetDirectories(folder, "*", SearchOption.AllDirectories).OrderByDescending(d => d.Length))
            {
                if (Directory.GetFileSystemEntries(dir).Length == 0)
                {
                    Directory.Delete(dir);
                    var meta = dir + ".meta";
                    if (File.Exists(meta)) File.Delete(meta);
                }
            }
            foreach (var file in files)
            {
                var path = Path.Combine(folder, file.Key);
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                if (File.Exists(path) && File.ReadAllBytes(path).AsSpan().SequenceEqual(file.Value)) continue;
                WriteFile(path, file.Value);
            }
            if (group == LibraryGroup.IOS) SignForSimulator(folder, libraries);
        }

        private static string Normalize(string path) => path.Replace('\\', '/');

        private static void Delete(string path)
        {
            try
            {
                File.Delete(path);
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
            {
                throw new ReleaseException($"Cannot remove {path}: the Editor has loaded it. Restart Unity, and Morflow will update the libraries when the project opens.");
            }
        }

        private static void WriteFile(string path, byte[] bytes)
        {
            try
            {
                File.WriteAllBytes(path, bytes);
            }
            catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
            {
                throw new ReleaseException($"Cannot replace {path}: the Editor has loaded it. Restart Unity, and Morflow will update the libraries when the project opens.");
            }
        }

        /// <summary>
        /// Signs the changed iOS libraries ad hoc on a Mac, as
        /// install_name_tool users do. Xcode signs them again with the app's
        /// identity when it embeds them; this keeps simulator builds working.
        /// </summary>
        private static void SignForSimulator(string folder, List<Library> libraries)
        {
            if (Application.platform != RuntimePlatform.OSXEditor) return;
            foreach (var library in libraries)
            {
                var binary = Path.Combine(folder, $"{library.Stem}.framework", library.Stem);
                var process = Process.Start(new ProcessStartInfo("codesign", $"--force --sign - \"{binary}\"")
                {
                    UseShellExecute = false,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                });
                var error = process.StandardError.ReadToEnd();
                process.WaitForExit();
                if (process.ExitCode != 0) throw new ReleaseException($"codesign failed for {binary}: {error}");
            }
        }

        private static string InfoPlist(string stem, string version, string platform) => $@"<?xml version=""1.0"" encoding=""UTF-8""?>
<!DOCTYPE plist PUBLIC ""-//Apple//DTD PLIST 1.0//EN"" ""http://www.apple.com/DTDs/PropertyList-1.0.dtd"">
<plist version=""1.0"">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>{stem}</string>
  <key>CFBundleIdentifier</key><string>org.morflow.{stem.Replace('_', '-')}</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>{stem}</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
  <key>CFBundleVersion</key><string>{version}</string>
  <key>CFBundleSupportedPlatforms</key><array><string>{platform}</string></array>
  <key>MinimumOSVersion</key><string>13.0</string>
</dict>
</plist>
";

        private static void WriteGitignore(string root)
        {
            var path = Path.Combine(root, ".gitignore");
            const string text = "# Downloaded by the Morflow package from ProjectSettings/Morflow.json; not committed.\n*\n!.gitignore\n";
            if (!File.Exists(path) || File.ReadAllText(path) != text) File.WriteAllText(path, text);
        }

        /// <summary>Sets where Unity loads and ships the library at <paramref name="assetPath"/>.</summary>
        private static void Configure(LibraryGroup group, string assetPath)
        {
            if (!(AssetImporter.GetAtPath(assetPath) is PluginImporter importer))
            {
                throw new ReleaseException($"Unity did not import {assetPath} as a plugin");
            }
            importer.ClearSettings();
            importer.SetCompatibleWithAnyPlatform(false);
            importer.SetCompatibleWithEditor(false);
            switch (group)
            {
                case LibraryGroup.Linux:
                    Desktop(importer, "Linux", "x86_64", BuildTarget.StandaloneLinux64);
                    break;
                case LibraryGroup.Windows:
                    Desktop(importer, "Windows", "x86_64", BuildTarget.StandaloneWindows64);
                    break;
                case LibraryGroup.MacOS:
                    Desktop(importer, "OSX", "AnyCPU", BuildTarget.StandaloneOSX);
                    break;
                case LibraryGroup.Android:
                    importer.SetCompatibleWithPlatform(BuildTarget.Android, true);
                    importer.SetPlatformData(BuildTarget.Android, "CPU", assetPath.Contains("/arm64-v8a/") ? "ARM64" : "X86_64");
                    break;
                case LibraryGroup.IOS:
                    importer.SetCompatibleWithPlatform(BuildTarget.iOS, true);
                    importer.SetPlatformData(BuildTarget.iOS, "AddToEmbeddedBinaries", "true");
                    break;
            }
            importer.SaveAndReimport();
        }

        private static void Desktop(PluginImporter importer, string os, string cpu, BuildTarget target)
        {
            importer.SetCompatibleWithEditor(true);
            importer.SetEditorData("OS", os);
            importer.SetEditorData("CPU", cpu);
            importer.SetCompatibleWithPlatform(target, true);
            importer.SetPlatformData(target, "CPU", cpu);
        }
    }
}
