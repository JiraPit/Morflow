using System;

namespace MorflowUnity
{
    /// <summary>
    /// Compiles Morflow pipelines.
    /// </summary>
    /// <example>
    /// <code>
    /// public MorflowPipelineAsset clean;
    ///
    /// async void Start()
    /// {
    ///     using var pipeline = Morflow.FromAsset(clean);
    ///     var wav = await pipeline.RunAsync&lt;MorflowTensor&gt;(File.ReadAllBytes(path));
    ///     File.WriteAllBytes(outPath, wav.ToBytes());
    /// }
    /// </code>
    /// </example>
    public static class Morflow
    {
        /// <summary>The engine release this package runs.</summary>
        public const string EngineVersion = "0.3.5";

        /// <summary>This package's version.</summary>
        public const string PackageVersion = "0.1.0";

        private static bool versionChecked;

        /// <summary>Compiles a <c>.morf</c> file from the project's Assets.</summary>
        public static MorflowPipeline FromAsset(MorflowPipelineAsset asset)
        {
            if (asset == null) throw new ArgumentNullException(nameof(asset), "No pipeline asset is assigned");
            return FromStr(asset.Text);
        }

        /// <summary>Compiles pipeline source text.</summary>
        public static MorflowPipeline FromStr(string source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            return Compile(Native.FromStr, source);
        }

        /// <summary>Same as <see cref="FromStr"/>.</summary>
        public static MorflowPipeline FromString(string source) => FromStr(source);

        /// <summary>
        /// Compiles a <c>.morf</c> file from disk, such as one in
        /// <c>Application.streamingAssetsPath</c> on desktop platforms.
        /// </summary>
        public static MorflowPipeline Load(string path)
        {
            if (path == null) throw new ArgumentNullException(nameof(path));
            return Compile(Native.Load, path);
        }

        private delegate IntPtr CompileFn(byte[] text, out IntPtr error);

        private static MorflowPipeline Compile(CompileFn compile, string text)
        {
            CheckVersion();
            var handle = compile(Native.Utf8(text), out var error);
            if (handle == IntPtr.Zero) throw new MorflowException(Native.TakeString(error) ?? "The pipeline could not be compiled");
            return new MorflowPipeline(handle);
        }

        /// <summary>Fails clearly when the native library and this package disagree.</summary>
        private static void CheckVersion()
        {
            if (versionChecked) return;
            PacksPath.Configure();
            string native;
            try
            {
                native = Native.ReadUtf8(Native.Version());
            }
            catch (DllNotFoundException e)
            {
                throw new MorflowException(
                    "The Morflow engine library is missing. In the Editor it is downloaded into Assets/Plugins/Morflow when the project opens; check the Console for a download error, or use Tools > Morflow > Refresh Native Libraries.",
                    e);
            }
            if (native != EngineVersion)
            {
                throw new MorflowException(
                    $"The Morflow engine library is version {native}, but this package runs engine {EngineVersion}. Restart Unity so it loads the new library.");
            }
            versionChecked = true;
        }
    }
}
