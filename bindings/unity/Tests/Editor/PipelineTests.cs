// Runs pipelines with the engine and packs the Editor loads from
// Assets/Plugins/Morflow. The project must list base, audio_basics,
// math_basics and nn_basics in ProjectSettings/Morflow.json, as the example
// project in Example~ does.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MorflowUnity.Editor;
using NUnit.Framework;

namespace MorflowUnity.Tests
{
    public class PipelineTests
    {
        private const string Identity = "accept Tensor $t\n$t >> base/latest/identity >> emit";

        [OneTimeSetUp]
        public void PrepareLibraries()
        {
            var group = NativeLibraries.EditorGroup();
            if (group == null) Assert.Ignore("Morflow has no libraries for this Editor's platform");
            NativeLibraries.Sync(new[] { group.Value }, Settings.Read(Hooks.ProjectRoot), Hooks.ProjectRoot);
        }

        [Test]
        public void RunsOnThisThreadAndOnAWorker()
        {
            using (var pipeline = Morflow.FromStr(Identity))
            {
                var input = new[] { 1f, 2.5f, -3f };
                Assert.AreEqual(input, pipeline.Run<MorflowTensor>(input).ToFloatArray());
                Assert.AreEqual(input, pipeline.RunAsync<MorflowTensor>(input).Result.ToFloatArray());
                Assert.AreEqual(new[] { "t" }, pipeline.Params.ToArray());
            }
        }

        [Test]
        public void KeepsTheInputDType()
        {
            using (var pipeline = Morflow.FromStr(Identity))
            {
                var output = pipeline.Run<MorflowTensor>(new TensorInput(new double[] { -4, 5 }, 2));
                Assert.AreEqual("f64", output.DType);
                Assert.AreEqual(new double[] { -4, 5 }, output.ToDoubleArray());
                Assert.AreEqual(new long[] { 2 }, output.Shape.ToArray());
                // An output goes back in as an input.
                Assert.AreEqual(new double[] { -4, 5 }, pipeline.Run<MorflowTensor>(output).ToDoubleArray());
            }
        }

        [Test]
        public void SelectsTheBundledVersion()
        {
            using (var pipeline = Morflow.FromStr("from base/0.3.4 import identity\naccept Tensor $t\n$t >> identity >> emit"))
            {
                Assert.AreEqual(new[] { 4f }, pipeline.Run<MorflowTensor>(new[] { 4f }).ToFloatArray());
            }
        }

        [Test]
        public void TurnsAudioIntoAWavFile()
        {
            using (var pipeline = Morflow.FromStr("import audio_basics/latest\naccept Audio $audio\n$audio >> gain(linear=2.0) >> to_wav >> emit"))
            {
                var samples = Enumerable.Range(0, 2205).Select(i => 0.25f * (float)Math.Sin(i / 10.0)).ToArray();
                var wav = pipeline.Run<MorflowTensor>(TensorInput.Audio(samples, 1, 22050)).ToBytes();
                Assert.AreEqual("RIFF", Encoding.ASCII.GetString(wav, 0, 4));
                Assert.AreEqual("WAVE", Encoding.ASCII.GetString(wav, 8, 4));
                // Interleaved stereo from AudioClip.GetData works too.
                var stereo = pipeline.Run<MorflowTensor>(TensorInput.Audio(new float[200], 2, 44100, interleaved: true)).ToBytes();
                Assert.AreEqual("RIFF", Encoding.ASCII.GetString(stereo, 0, 4));
            }
        }

        [Test]
        public void AddsACompositePair()
        {
            using (var pipeline = Morflow.FromStr("import math_basics/latest\naccept Composite $pair\n$pair >> add >> emit"))
            {
                var output = pipeline.Run<MorflowTensor>(new object[]
                {
                    new TensorInput(new[] { 1, 2, 3, 4, 5, 6 }, 2, 3),
                    new[] { 10, 20, 30 },
                });
                Assert.AreEqual(new[] { 11, 22, 33, 14, 25, 36 }, output.ToIntArray());
            }
        }

        [Test]
        public void TakesPositionalArguments()
        {
            using (var pipeline = Morflow.FromStr("import nn_basics/latest\naccept Scalar $value\n$value >> relu >> emit"))
            {
                Assert.AreEqual(new[] { 0f }, ((MorflowTensor)pipeline.RunArgs(-3.0)).ToFloatArray());
                Assert.AreEqual(new[] { 2f }, ((MorflowTensor)pipeline.RunArgsAsync(2).Result).ToFloatArray());
            }
        }

        [Test]
        public void ReturnsNamedOutputsAndComposites()
        {
            using (var pipeline = Morflow.FromStr("accept Tensor $t\n$t >> base/latest/identity >> emit(\"a\")\n$t >> emit(\"b\")"))
            {
                var outputs = pipeline.RunToMap(new[] { 7f });
                Assert.AreEqual(new[] { "a", "b" }, outputs.Keys.OrderBy(k => k).ToArray());
                Assert.AreEqual(new[] { 7f }, ((MorflowTensor)outputs["a"]).ToFloatArray());
            }
            using (var pipeline = Morflow.FromStr("accept Composite $pair\n$pair >> emit"))
            {
                var parts = pipeline.Run<IReadOnlyList<object>>(new object[] { new[] { 1.5 }, new long[] { 7 } });
                Assert.AreEqual(new[] { 1.5 }, ((MorflowTensor)parts[0]).ToDoubleArray());
                Assert.AreEqual(new long[] { 7 }, ((MorflowTensor)parts[1]).ToLongArray());
            }
        }

        [Test]
        public void ReportsErrors()
        {
            StringAssert.Contains("no_such_action", Assert.Throws<MorflowException>(() => Morflow.FromStr("accept Tensor $t\n$t >> no_such_action >> emit")).Message);
            Assert.Throws<ArgumentException>(() => Morflow.FromStr(Identity).Run(new object()));
            var pipeline = Morflow.FromStr(Identity);
            pipeline.Dispose();
            Assert.Throws<ObjectDisposedException>(() => pipeline.Run(new[] { 1f }));
            StringAssert.Contains("not MorflowTensor", Assert.Throws<InvalidCastException>(
                () => Morflow.FromStr("accept Composite $pair\n$pair >> emit").Run<MorflowTensor>(new object[] { new[] { 1f } })).Message);
        }

        [Test]
        public void ManyRunsAtOnce()
        {
            using (var pipeline = Morflow.FromStr(Identity))
            {
                var runs = Enumerable.Range(0, 50).Select(i => pipeline.RunAsync<MorflowTensor>(new[] { (float)i })).ToArray();
                Task.WaitAll(runs);
                for (var i = 0; i < runs.Length; i++) Assert.AreEqual(new[] { (float)i }, runs[i].Result.ToFloatArray());
            }
        }

        [Test]
        public void CompilesAnAsset()
        {
            var asset = MorflowPipelineAsset.Create(Identity);
            using (var pipeline = Morflow.FromAsset(asset)) Assert.AreEqual(new[] { "t" }, pipeline.Params.ToArray());
            Assert.Throws<ArgumentNullException>(() => Morflow.FromAsset(null));
        }
    }
}
