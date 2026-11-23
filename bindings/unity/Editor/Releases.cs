// Downloads Morflow's native libraries from its GitHub releases:
//   - the C interface library, from the engine release v<version>;
//   - pack libraries (every action of a pack in one library), from the pack's
//     release action_packs/<pack>/v<version>.
// Every file is checked against the release's checksums.txt and kept in
// Library/Morflow/releases, so later imports and builds work offline. The
// same logic as the Dart build hook and react-native-morflow.
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace MorflowUnity.Editor
{
    internal sealed class ReleaseException : Exception
    {
        public ReleaseException(string message) : base(message)
        {
        }
    }

    internal sealed class Releases : IDisposable
    {
        internal const string DefaultUrl = "https://github.com/JiraPit/Morflow/releases/download";

        private static readonly Regex ChecksumLine = new Regex(@"^([0-9a-f]{64}) [ *]?(.+)$");

        private readonly string baseUrl;
        private readonly string cache;
        private HttpClient client;

        /// <param name="baseUrl">GitHub's download URL (when null), a mirror, or a file: folder with the same &lt;tag&gt;/&lt;file&gt; layout.</param>
        /// <param name="cache">Where downloaded files are kept.</param>
        public Releases(string baseUrl, string cache)
        {
            this.baseUrl = (string.IsNullOrEmpty(baseUrl) ? DefaultUrl : baseUrl).TrimEnd('/');
            this.cache = cache;
        }

        public static string EngineTag(string version) => $"v{version}";
        public static string PackTag(string pack, string version) => $"action_packs/{pack}/v{version}";

        public static string Extension(string platform) =>
            platform.StartsWith("windows-", StringComparison.Ordinal) ? "dll"
            : platform.StartsWith("darwin-", StringComparison.Ordinal) || platform.StartsWith("ios-", StringComparison.Ordinal) ? "dylib"
            : "so";

        /// <summary>The file name the operating system loads, such as libmorflow_capi.so.</summary>
        public static string LibraryFile(string stem, string platform) =>
            $"{(platform.StartsWith("windows-", StringComparison.Ordinal) ? "" : "lib")}{stem}.{Extension(platform)}";

        public static string CapiAsset(string platform) => $"libmorflow_capi-{platform}.{Extension(platform)}";

        public static string PackAsset(string pack, string version, string platform) =>
            $"morflow_pack_{pack}-{version}-{platform}.{Extension(platform)}";

        /// <summary>Parses a checksums.txt written by sha256sum.</summary>
        public static Dictionary<string, string> ParseChecksums(string text)
        {
            var sums = new Dictionary<string, string>();
            foreach (var line in text.Split('\n'))
            {
                var match = ChecksumLine.Match(line.Trim());
                if (match.Success) sums[match.Groups[2].Value] = match.Groups[1].Value;
            }
            return sums;
        }

        /// <summary>
        /// The path of <paramref name="asset"/> from release <paramref name="tag"/>,
        /// downloaded when it is not cached yet. A cached copy is reused while it
        /// matches the release's checksum.
        /// </summary>
        public string Fetch(string tag, string asset)
        {
            var dir = Path.Combine(cache, Safe(tag));
            var file = Path.Combine(dir, Safe(asset));
            var checksumsFile = Path.Combine(dir, "checksums.txt");

            // Published files never change, so a cached checksum stays valid.
            if (File.Exists(checksumsFile) && File.Exists(file))
            {
                ParseChecksums(File.ReadAllText(checksumsFile)).TryGetValue(asset, out var cached);
                if (cached != null && Sha256(File.ReadAllBytes(file)) == cached) return file;
            }

            var checksumsText = Encoding.UTF8.GetString(Download(tag, "checksums.txt", $"release {tag}"));
            if (!ParseChecksums(checksumsText).TryGetValue(asset, out var expected))
            {
                throw new ReleaseException($"Morflow release {tag} has no {asset} ({Url(tag, "")}).");
            }
            Directory.CreateDirectory(dir);
            File.WriteAllText(checksumsFile, checksumsText);
            if (File.Exists(file) && Sha256(File.ReadAllBytes(file)) == expected) return file;

            var bytes = Download(tag, asset, $"{asset} from release {tag}");
            var actual = Sha256(bytes);
            if (actual != expected)
            {
                throw new ReleaseException(
                    $"{asset} from Morflow release {tag} does not match its checksum (expected {expected}, got {actual}). The download was corrupted or changed; it was not used.");
            }
            var partial = file + ".partial";
            File.WriteAllBytes(partial, bytes);
            if (File.Exists(file)) File.Delete(file);
            File.Move(partial, file);
            return file;
        }

        public void Dispose() => client?.Dispose();

        private string Url(string tag, string asset) => $"{baseUrl}/{tag}/{asset}";

        private byte[] Download(string tag, string asset, string what)
        {
            var url = Url(tag, asset);
            var web = url.StartsWith("http://", StringComparison.Ordinal) || url.StartsWith("https://", StringComparison.Ordinal);
            if (!web)
            {
                // A file: URL or a plain folder path.
                var path = url.StartsWith("file:", StringComparison.Ordinal) ? new Uri(url).LocalPath : Path.GetFullPath(url);
                if (!File.Exists(path)) throw new ReleaseException($"Could not find {what}: {path} does not exist.");
                return File.ReadAllBytes(path);
            }
            if (client == null)
            {
                client = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
                client.DefaultRequestHeaders.UserAgent.ParseAdd("Morflow-Unity");
            }
            try
            {
                using (var response = client.GetAsync(url).GetAwaiter().GetResult())
                {
                    if ((int)response.StatusCode == 404)
                    {
                        throw new ReleaseException($"Could not find {what}: {url} returned 404. Is the version published?");
                    }
                    if (!response.IsSuccessStatusCode)
                    {
                        throw new ReleaseException($"Could not download {what}: {url} returned HTTP {(int)response.StatusCode}.");
                    }
                    return response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult();
                }
            }
            catch (HttpRequestException e)
            {
                throw new ReleaseException($"Could not download {what} from {url}: {e.Message}");
            }
            catch (System.Threading.Tasks.TaskCanceledException)
            {
                throw new ReleaseException($"Downloading {what} from {url} timed out.");
            }
        }

        internal static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(bytes);
                var text = new StringBuilder(hash.Length * 2);
                foreach (var b in hash) text.Append(b.ToString("x2"));
                return text.ToString();
            }
        }

        // Tags such as action_packs/base/v0.3.4 become one folder name.
        private static string Safe(string name) => Regex.Replace(name, "[^A-Za-z0-9._-]", "_");
    }
}
