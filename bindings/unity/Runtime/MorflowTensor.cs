using System;
using System.Collections.Generic;
using System.Linq;

namespace MorflowUnity
{
    /// <summary>
    /// An emitted tensor. A Composite output is a list of outputs instead.
    /// </summary>
    public sealed class MorflowTensor
    {
        private readonly byte[] data;

        /// <summary>The dimensions, outermost first.</summary>
        public IReadOnlyList<long> Shape { get; }

        /// <summary>The element type, such as <c>"f32"</c>.</summary>
        public string DType { get; }

        internal MorflowTensor(long[] shape, string dtype, byte[] data)
        {
            Shape = Array.AsReadOnly(shape);
            DType = dtype;
            this.data = data;
        }

        /// <summary>The number of dimensions.</summary>
        public int Rank => Shape.Count;

        /// <summary>The number of elements.</summary>
        public long Length => Shape.Aggregate(1L, (count, dim) => count * dim);

        /// <summary>The elements as the array type matching <see cref="DType"/>.</summary>
        public Array ToArray()
        {
            switch (DType)
            {
                case "f32": return ToFloatArray();
                case "f64": return ToDoubleArray();
                case "u8": return ToByteArray();
                case "i8": return ToSByteArray();
                case "i16": return ToShortArray();
                case "i32": return ToIntArray();
                case "i64": return ToLongArray();
                case "u32": return ToUIntArray();
                case "u64": return ToULongArray();
                default: throw new NotSupportedException($"No array type for dtype '{DType}'");
            }
        }

        public float[] ToFloatArray() => Elements<float>(4);
        public double[] ToDoubleArray() => Elements<double>(8);
        public byte[] ToByteArray() => (byte[])data.Clone();
        public sbyte[] ToSByteArray() => Elements<sbyte>(1);
        public short[] ToShortArray() => Elements<short>(2);
        public int[] ToIntArray() => Elements<int>(4);
        public long[] ToLongArray() => Elements<long>(8);
        public uint[] ToUIntArray() => Elements<uint>(4);
        public ulong[] ToULongArray() => Elements<ulong>(8);

        /// <summary>
        /// The raw element bytes, in native byte order. For bytes a pipeline
        /// emits, such as a WAV file from <c>to_wav</c>, this is the file.
        /// </summary>
        public byte[] ToBytes() => (byte[])data.Clone();

        /// <summary>The raw bytes without a copy. Do not change them.</summary>
        internal byte[] RawData => data;

        private T[] Elements<T>(int size) where T : struct
        {
            var elements = new T[data.Length / size];
            Buffer.BlockCopy(data, 0, elements, 0, elements.Length * size);
            return elements;
        }

        public override string ToString() => $"MorflowTensor(dtype: {DType}, shape: [{string.Join(", ", Shape)}])";
    }
}
