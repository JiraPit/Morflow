// Tests that need no engine: settings, release downloads and Mach-O edits.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MorflowUnity.Editor;
using NUnit.Framework;

namespace MorflowUnity.Tests
{
    public class SettingsTests
    {
        [Test]
        public void ReadsPacksAndReleaseUrl()
        {
            var settings = Settings.Parse("{\n  \"packs\": {\"base\": \"0.3.4\", \"audio_basics\": \"0.2.5\"},\n  \"releaseUrl\": \"file:///tmp/r\"\n}");
            Assert.AreEqual(new[] { "audio_basics", "base" }, settings.Packs.Keys.ToArray());
            Assert.AreEqual("0.2.5", settings.Packs["audio_basics"]);
            Assert.AreEqual("file:///tmp/r", settings.ReleaseUrl);
            Assert.IsEmpty(Settings.Parse("{}").Packs);
        }

        [Test]
        public void ReportsWhatItCannotRead()
        {
            StringAssert.Contains("Cannot read", Assert.Throws<SettingsException>(() => Settings.Parse("{")).Message);
            StringAssert.Contains("must be an object", Assert.Throws<SettingsException>(() => Settings.Parse("{\"packs\": []}")).Message);
            StringAssert.Contains("exact version", Assert.Throws<SettingsException>(() => Settings.Parse("{\"packs\": {\"base\": \"latest\"}}")).Message);
            StringAssert.Contains("invalid pack name", Assert.Throws<SettingsException>(() => Settings.Parse("{\"packs\": {\"Base\": \"1.0.0\"}}")).Message);
        }
    }

    public class ReleasesTests
    {
        private string root;

        [SetUp]
        public void SetUp()
        {
            root = Path.Combine(Path.GetTempPath(), "morflow-unity-test-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
        }

        [TearDown]
        public void TearDown() => Directory.Delete(root, true);

        private string Publish(string tag, string asset, byte[] bytes, string checksum = null)
        {
            var dir = Path.Combine(root, "releases", tag);
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, asset), bytes);
            File.WriteAllText(Path.Combine(dir, "checksums.txt"), $"{checksum ?? Releases.Sha256(bytes)}  {asset}\n");
            return new Uri(Path.Combine(root, "releases")).AbsoluteUri;
        }

        [Test]
        public void NamesFilesLikeTheReleases()
        {
            Assert.AreEqual("libmorflow_capi-android-aarch64.so", Releases.CapiAsset("android-aarch64"));
            Assert.AreEqual("morflow_pack_base-0.3.4-windows-x86_64.dll", Releases.PackAsset("base", "0.3.4", "windows-x86_64"));
            Assert.AreEqual("morflow_pack_base.dll", Releases.LibraryFile("morflow_pack_base", "windows-x86_64"));
            Assert.AreEqual("libmorflow_capi.dylib", Releases.LibraryFile("morflow_capi", "ios-arm64"));
            Assert.AreEqual("action_packs/base/v0.3.4", Releases.PackTag("base", "0.3.4"));
        }

        [Test]
        public void ParsesChecksums()
        {
            var a = new string('a', 64);
            var sums = Releases.ParseChecksums($"{a}  one.so\n{new string('b', 64)} *two.dll\nnot a line\n");
            Assert.AreEqual(a, sums["one.so"]);
            Assert.AreEqual(2, sums.Count);
        }

        [Test]
        public void FetchesChecksAndCaches()
        {
            var bytes = Encoding.UTF8.GetBytes("library");
            var url = Publish("v1.0.0", "lib.so", bytes);
            using (var releases = new Releases(url, Path.Combine(root, "cache")))
            {
                var file = releases.Fetch("v1.0.0", "lib.so");
                Assert.AreEqual(bytes, File.ReadAllBytes(file));
                // Cached: works with the release gone.
                Directory.Delete(Path.Combine(root, "releases"), true);
                Assert.AreEqual(file, releases.Fetch("v1.0.0", "lib.so"));
            }
        }

        [Test]
        public void RejectsBadDownloadsAndMissingFiles()
        {
            var url = Publish("action_packs/base/v0.3.4", "lib.so", Encoding.UTF8.GetBytes("changed"), new string('0', 64));
            using (var releases = new Releases(url, Path.Combine(root, "cache")))
            {
                StringAssert.Contains("does not match its checksum", Assert.Throws<ReleaseException>(() => releases.Fetch("action_packs/base/v0.3.4", "lib.so")).Message);
                StringAssert.Contains("has no other.so", Assert.Throws<ReleaseException>(() => releases.Fetch("action_packs/base/v0.3.4", "other.so")).Message);
                StringAssert.Contains("does not exist", Assert.Throws<ReleaseException>(() => releases.Fetch("v9.9.9", "lib.so")).Message);
            }
        }
    }

    public class MachOTests
    {
        /// <summary>A minimal thin 64-bit Mach-O library with one LC_ID_DYLIB.</summary>
        private static byte[] Library(uint cpu, string installName, int size = 200)
        {
            var bytes = new byte[size];
            void Put(int at, uint value) => BitConverter.GetBytes(value).CopyTo(bytes, at);
            Put(0, 0xfeedfacf);
            Put(4, cpu);
            Put(8, 0);
            Put(12, 6); // MH_DYLIB
            Put(16, 1); // ncmds
            const int commandSize = 120;
            Put(20, commandSize);
            Put(32, 0xd);
            Put(36, commandSize);
            Put(40, 24);
            Encoding.UTF8.GetBytes(installName).CopyTo(bytes, 32 + 24);
            return bytes;
        }

        [Test]
        public void SetsTheInstallName()
        {
            var library = Library(0x0100000c, "/Users/runner/work/Morflow/target/aarch64-apple-ios/release/deps/libmorflow_capi.dylib");
            MachO.SetInstallName(library, "@rpath/morflow_capi.framework/morflow_capi");
            Assert.AreEqual("@rpath/morflow_capi.framework/morflow_capi", MachO.InstallName(library));
        }

        [Test]
        public void RefusesANameThatDoesNotFit()
        {
            var library = Library(0x0100000c, "short");
            Assert.Throws<InvalidDataException>(() => MachO.SetInstallName(library, "@rpath/" + new string('x', 200)));
        }

        [Test]
        public void JoinsSlicesIntoAUniversalBinary()
        {
            var arm = Library(0x0100000c, "arm");
            var intel = Library(0x01000007, "intel");
            var fat = MachO.Universal(new List<byte[]> { arm, intel });
            Assert.AreEqual(new uint[] { 0x0100000c, 0x01000007 }, MachO.UniversalCpuTypes(fat).ToArray());
            // The arm64 slice starts on a 16 KB boundary and is intact.
            var offset = (fat[16] << 24) | (fat[17] << 16) | (fat[18] << 8) | fat[19];
            Assert.AreEqual(16384, offset);
            Assert.AreEqual(arm, fat.Skip(offset).Take(arm.Length).ToArray());
        }
    }
}
