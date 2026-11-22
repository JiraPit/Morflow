using System;

namespace MorflowUnity
{
    /// <summary>
    /// A tensor, image or audio input with an explicit shape.
    /// </summary>
    /// <remarks>
    /// <see cref="Data"/> is an array of elements: <c>float[]</c>, <c>double[]</c>,
    /// <c>byte[]</c>, <c>sbyte[]</c>, <c>short[]</c>, <c>int[]</c>, <c>long[]</c>,
    /// <c>uint[]</c> or <c>ulong[]</c>. When <see cref="DType"/> is null it is
    /// inferred from the array (a <c>double[]</c> is <c>"f64"</c>), except for a
    /// <c>byte[]</c>, which keeps the <c>"f32"</c> default unless you say
    /// <c>"u8"</c>. A <c>byte[]</c> can also hold the raw bytes of any dtype.
    /// </remarks>
    public sealed class TensorInput
    {
        /// <summary>The elements, as an array of a numeric type.</summary>
        public Array Data { get; set; }

        /// <summary>The dimensions, outermost first.</summary>
        public long[] Shape { get; set; }

        /// <summary>
        /// Element type: <c>"f32"</c>, <c>"f64"</c>, <c>"u8"</c>, <c>"i8"</c>,
        /// <c>"i16"</c>, <c>"i32"</c>, <c>"i64"</c>, <c>"u32"</c> or <c>"u64"</c>.
        /// </summary>
        public string DType { get; set; }

        /// <summary>
        /// The payload type to send: <c>"tensor"</c> (the default), <c>"image"</c>
        /// or <c>"audio"</c>. There is no inference: a rank-3 array is a plain
        /// tensor unless you ask for <c>"image"</c>.
        /// </summary>
        public string PayloadType { get; set; }

        /// <summary>
        /// Color space for images: <c>"grayscale"</c>, <c>"rgb"</c>, <c>"rgba"</c>,
        /// <c>"bgr"</c> or <c>"bgra"</c>. Inferred from the channel count when null.
        /// </summary>
        public string ColorSpace { get; set; }

        /// <summary>Sample rate in Hz for audio. Defaults to 44100.</summary>
        public int? SampleRate { get; set; }

        /// <summary>Channel count for audio. Defaults to the leading dimension.</summary>
        public int? Channels { get; set; }

        /// <summary>
        /// Memory layout: <c>"hwc"</c> or <c>"chw"</c> for images, <c>"planar"</c>
        /// or <c>"interleaved"</c> for audio. Defaults to <c>"hwc"</c> and <c>"planar"</c>.
        /// </summary>
        public string Layout { get; set; }

        public TensorInput()
        {
        }

        public TensorInput(Array data, params long[] shape)
        {
            Data = data;
            Shape = shape;
        }

        /// <summary>An image with <paramref name="height"/> rows of <paramref name="width"/> pixels.</summary>
        public static TensorInput Image(Array data, int height, int width, int channels, string colorSpace = null) =>
            new TensorInput(data, height, width, channels) { PayloadType = "image", ColorSpace = colorSpace };

        /// <summary>
        /// Audio samples, one row per channel (planar). Unity's
        /// <c>AudioClip.GetData</c> gives interleaved samples: pass
        /// <paramref name="interleaved"/> as true for those.
        /// </summary>
        public static TensorInput Audio(float[] samples, int channels, int sampleRate, bool interleaved = false) =>
            new TensorInput(samples, interleaved ? new long[] { samples.Length / channels, channels } : new long[] { channels, samples.Length / channels })
            {
                PayloadType = "audio",
                SampleRate = sampleRate,
                Channels = channels,
                Layout = interleaved ? "interleaved" : "planar",
            };
    }
}
