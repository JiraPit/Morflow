// Converts C# values to native inputs and native results to C# values,
// following the same rules as the Node.js and Dart bindings.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MorflowUnity
{
    /// <summary>Native memory for one call, freed all at once.</summary>
    internal sealed class Arena : IDisposable
    {
        private readonly List<IntPtr> blocks = new List<IntPtr>();
        private readonly List<GCHandle> pins = new List<GCHandle>();

        public IntPtr Alloc(int bytes)
        {
            var block = Marshal.AllocHGlobal(Math.Max(bytes, 1));
            blocks.Add(block);
            return block;
        }

        /// <summary>The address of <paramref name="array"/>'s elements, pinned until disposal.</summary>
        public IntPtr Pin(Array array)
        {
            if (array.Length == 0) return IntPtr.Zero;
            var pin = GCHandle.Alloc(array, GCHandleType.Pinned);
            pins.Add(pin);
            return pin.AddrOfPinnedObject();
        }

        public IntPtr String(string text) => text == null ? IntPtr.Zero : Pin(Native.Utf8(text));

        public void Dispose()
        {
            foreach (var pin in pins) pin.Free();
            foreach (var block in blocks) Marshal.FreeHGlobal(block);
            pins.Clear();
            blocks.Clear();
        }
    }

    internal static class Convert
    {
        private static readonly int ValueInSize = Marshal.SizeOf<Native.ValueIn>();
        private static readonly int ValueOutSize = Marshal.SizeOf<Native.ValueOut>();

        private abstract class Value
        {
        }

        private sealed class TensorValue : Value
        {
            public Array Data;
            public long[] Shape;
            public string DType;
            public string PayloadType;
            public string ColorSpace;
            public string Layout;
            public int? SampleRate;
            public int? Channels;
            public List<TensorValue> Items = new List<TensorValue>();
        }

        private sealed class BytesValue : Value
        {
            public byte[] Bytes;
        }

        private sealed class NumberValue : Value
        {
            public double Number;
        }

        private sealed class TextValue : Value
        {
            public string Text;
        }

        private sealed class BoolValue : Value
        {
            public bool Flag;
        }

        /// <summary>The dtype an array's element type carries, or null (bytes, unsupported).</summary>
        internal static string ArrayDType(Array data)
        {
            switch (data)
            {
                case float[] _: return "f32";
                case double[] _: return "f64";
                case sbyte[] _: return "i8";
                case short[] _: return "i16";
                case int[] _: return "i32";
                case long[] _: return "i64";
                case uint[] _: return "u32";
                case ulong[] _: return "u64";
                default: return null;
            }
        }

        private static bool IsElementArray(Array data) => data is byte[] || ArrayDType(data) != null;

        /// <summary>
        /// A pipeline input: a <see cref="TensorInput"/>, a numeric array (a
        /// 1-D tensor of its dtype), a <c>byte[]</c> of raw bytes, a
        /// <see cref="MorflowTensor"/> from an earlier run, or a list of those,
        /// sent as a Composite.
        /// </summary>
        private static Value Input(object value)
        {
            switch (value)
            {
                case null:
                    throw new ArgumentNullException(nameof(value), "A pipeline input cannot be null");
                case byte[] bytes:
                    return new BytesValue { Bytes = bytes };
                case Array array when ArrayDType(array) != null:
                    if (array.Rank != 1) throw new ArgumentException("Pass a one-dimensional array, with the dimensions in a TensorInput's Shape");
                    return new TensorValue { Data = array, Shape = new long[] { array.Length }, DType = ArrayDType(array) };
                case TensorInput input:
                    if (input.Data == null || !IsElementArray(input.Data) || input.Data.Rank != 1)
                    {
                        throw new ArgumentException("TensorInput.Data must be a one-dimensional float[], double[], byte[], sbyte[], short[], int[], long[], uint[] or ulong[]");
                    }
                    if (input.Shape == null) throw new ArgumentException("TensorInput.Shape is required");
                    foreach (var dim in input.Shape)
                    {
                        if (dim < 0) throw new ArgumentException("TensorInput.Shape dimensions must not be negative");
                    }
                    return new TensorValue
                    {
                        Data = input.Data,
                        Shape = input.Shape,
                        DType = input.DType ?? ArrayDType(input.Data),
                        PayloadType = input.PayloadType,
                        ColorSpace = input.ColorSpace,
                        Layout = input.Layout,
                        SampleRate = input.SampleRate,
                        Channels = input.Channels,
                    };
                case MorflowTensor tensor:
                    return new TensorValue
                    {
                        Data = tensor.RawData,
                        Shape = new List<long>(tensor.Shape).ToArray(),
                        DType = tensor.DType,
                    };
                case IList list when !(value is Array array && IsElementArray(array)):
                    var composite = new TensorValue { Data = Array.Empty<byte>(), Shape = Array.Empty<long>(), PayloadType = "composite" };
                    foreach (var item in list) composite.Items.Add(Component(item));
                    return composite;
                default:
                    throw new ArgumentException(
                        $"Unsupported pipeline input {value.GetType().Name}; expected a TensorInput, a numeric array, a byte[] of raw bytes, a MorflowTensor or a list of those (Composite)");
            }
        }

        private static TensorValue Component(object value)
        {
            if (value != null && Input(value) is TensorValue tensor) return tensor;
            throw new ArgumentException("Composite components must be TensorInputs, numeric arrays (not byte[]), MorflowTensors or lists");
        }

        /// <summary>
        /// A positional parameter: anything <see cref="Input"/> accepts, a number
        /// (Scalar or <c>*Arg</c>), a string (<c>StrArg</c>) or a bool (<c>BoolArg</c>).
        /// </summary>
        private static Value Arg(object value)
        {
            switch (value)
            {
                case null: throw new ArgumentNullException(nameof(value), "A pipeline argument cannot be null");
                case bool flag: return new BoolValue { Flag = flag };
                case string text: return new TextValue { Text = text };
                case double _:
                case float _:
                case int _:
                case long _:
                case short _:
                case byte _:
                case sbyte _:
                case uint _:
                case ulong _:
                case ushort _:
                case decimal _:
                    return new NumberValue { Number = System.Convert.ToDouble(value) };
                default: return Input(value);
            }
        }

        private static void Write(IntPtr target, Value value, Arena arena)
        {
            var native = new Native.ValueIn { Kind = Native.ValueTensor, SampleRate = -1, Channels = -1 };
            switch (value)
            {
                case TensorValue tensor:
                    var shape = arena.Alloc(Math.Max(tensor.Shape.Length, 1) * 8);
                    for (var i = 0; i < tensor.Shape.Length; i++) Marshal.WriteInt64(shape, i * 8, tensor.Shape[i]);
                    native.Data = arena.Pin(tensor.Data);
                    native.DataLen = (UIntPtr)(ulong)Buffer.ByteLength(tensor.Data);
                    native.Shape = shape;
                    native.Rank = (UIntPtr)(ulong)tensor.Shape.Length;
                    native.DType = arena.String(tensor.DType);
                    native.PayloadType = arena.String(tensor.PayloadType);
                    native.ColorSpace = arena.String(tensor.ColorSpace);
                    native.Layout = arena.String(tensor.Layout);
                    native.SampleRate = tensor.SampleRate ?? -1;
                    native.Channels = tensor.Channels ?? -1;
                    if (tensor.Items.Count > 0)
                    {
                        var items = arena.Alloc(tensor.Items.Count * ValueInSize);
                        for (var i = 0; i < tensor.Items.Count; i++) Write(items + i * ValueInSize, tensor.Items[i], arena);
                        native.Items = items;
                        native.ItemCount = (UIntPtr)(ulong)tensor.Items.Count;
                    }
                    break;
                case BytesValue bytes:
                    native.Kind = Native.ValueBytes;
                    native.Data = arena.Pin(bytes.Bytes);
                    native.DataLen = (UIntPtr)(ulong)bytes.Bytes.Length;
                    break;
                case NumberValue number:
                    native.Kind = Native.ValueNumber;
                    native.Number = number.Number;
                    break;
                case TextValue text:
                    var utf8 = Encoding.UTF8.GetBytes(text.Text);
                    native.Kind = Native.ValueText;
                    native.Data = arena.Pin(utf8);
                    native.DataLen = (UIntPtr)(ulong)utf8.Length;
                    break;
                case BoolValue flag:
                    native.Kind = Native.ValueBool;
                    native.Flag = flag.Flag ? (byte)1 : (byte)0;
                    break;
            }
            Marshal.StructureToPtr(native, target, false);
        }

        /// <summary>The native form of a single-input run's input (zero for none).</summary>
        internal static IntPtr EncodeInput(object input, Arena arena)
        {
            if (input == null) return IntPtr.Zero;
            var value = Input(input);
            var target = arena.Alloc(ValueInSize);
            Write(target, value, arena);
            return target;
        }

        /// <summary>The native form of positional parameters, in declaration order.</summary>
        internal static IntPtr EncodeArgs(object[] args, Arena arena)
        {
            if (args == null || args.Length == 0) return IntPtr.Zero;
            var values = new Value[args.Length];
            for (var i = 0; i < args.Length; i++) values[i] = Arg(args[i]);
            var list = arena.Alloc(values.Length * ValueInSize);
            for (var i = 0; i < values.Length; i++) Write(list + i * ValueInSize, values[i], arena);
            return list;
        }

        private static object ReadValue(IntPtr pointer)
        {
            var value = Marshal.PtrToStructure<Native.ValueOut>(pointer);
            var dtype = Native.ReadUtf8(value.DType);
            if (dtype == "composite")
            {
                var items = new List<object>();
                for (var i = 0; i < (int)(ulong)value.ItemCount; i++) items.Add(ReadValue(value.Items + i * ValueOutSize));
                return items.AsReadOnly();
            }
            var rank = (int)(ulong)value.Rank;
            var shape = new long[rank];
            for (var i = 0; i < rank; i++) shape[i] = Marshal.ReadInt64(value.Shape, i * 8);
            // Copied out of native memory, which is freed with the result.
            var data = new byte[(int)(ulong)value.DataLen];
            if (data.Length > 0) Marshal.Copy(value.Data, data, 0, data.Length);
            return new MorflowTensor(shape, dtype, data);
        }

        /// <summary>Reads and frees a result: the single output, or the named outputs.</summary>
        internal static object DecodeResult(IntPtr result, bool all)
        {
            try
            {
                var error = Native.ResultError(result);
                if (error != IntPtr.Zero) throw new MorflowException(Native.ReadUtf8(error));
                if (!all) return ReadValue(Native.ResultValue(result, UIntPtr.Zero));
                var outputs = new Dictionary<string, object>();
                var count = (ulong)Native.ResultCount(result);
                for (ulong i = 0; i < count; i++)
                {
                    outputs[Native.ReadUtf8(Native.ResultName(result, (UIntPtr)i))] = ReadValue(Native.ResultValue(result, (UIntPtr)i));
                }
                return outputs;
            }
            finally
            {
                Native.ResultFree(result);
            }
        }
    }
}
