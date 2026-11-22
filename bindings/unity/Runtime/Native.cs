// The C interface to the Morflow engine (bindings/capi/src/lib.rs). Only
// this package calls it; applications use Morflow and MorflowPipeline.
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace MorflowUnity
{
    internal static class Native
    {
#if UNITY_IOS && !UNITY_EDITOR
        // On iOS the engine is an embedded framework the app links against.
        private const string Library = "__Internal";
#else
        // libmorflow_capi.so / .dylib, or morflow_capi.dll on Windows.
        private const string Library = "morflow_capi";
#endif

        internal const uint ValueTensor = 0;
        internal const uint ValueBytes = 1;
        internal const uint ValueNumber = 2;
        internal const uint ValueText = 3;
        internal const uint ValueBool = 4;

        /// <summary>MorflowValueIn: a value passed into a pipeline.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct ValueIn
        {
            public uint Kind;
            public IntPtr Data;
            public UIntPtr DataLen;
            public IntPtr Shape;
            public UIntPtr Rank;
            public IntPtr DType;
            public IntPtr PayloadType;
            public IntPtr ColorSpace;
            public IntPtr Layout;
            public long SampleRate;
            public long Channels;
            public IntPtr Items;
            public UIntPtr ItemCount;
            public double Number;
            public byte Flag;
        }

        /// <summary>MorflowValueOut: an emitted value.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct ValueOut
        {
            public IntPtr DType;
            public IntPtr Shape;
            public UIntPtr Rank;
            public IntPtr Data;
            public UIntPtr DataLen;
            public IntPtr Items;
            public UIntPtr ItemCount;
        }

        [DllImport(Library, EntryPoint = "morflow_version", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr Version();

        [DllImport(Library, EntryPoint = "morflow_load", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr Load(byte[] path, out IntPtr error);

        [DllImport(Library, EntryPoint = "morflow_from_str", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr FromStr(byte[] source, out IntPtr error);

        [DllImport(Library, EntryPoint = "morflow_pipeline_free", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void PipelineFree(IntPtr pipeline);

        [DllImport(Library, EntryPoint = "morflow_param_count", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UIntPtr ParamCount(IntPtr pipeline);

        [DllImport(Library, EntryPoint = "morflow_param_name", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr ParamName(IntPtr pipeline, UIntPtr index);

        [DllImport(Library, EntryPoint = "morflow_string_free", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void StringFree(IntPtr text);

        [DllImport(Library, EntryPoint = "morflow_run", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr Run(IntPtr pipeline, IntPtr input, byte all);

        [DllImport(Library, EntryPoint = "morflow_run_args", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr RunArgs(IntPtr pipeline, IntPtr args, UIntPtr count, byte all);

        [DllImport(Library, EntryPoint = "morflow_result_error", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr ResultError(IntPtr result);

        [DllImport(Library, EntryPoint = "morflow_result_count", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UIntPtr ResultCount(IntPtr result);

        [DllImport(Library, EntryPoint = "morflow_result_name", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr ResultName(IntPtr result, UIntPtr index);

        [DllImport(Library, EntryPoint = "morflow_result_value", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr ResultValue(IntPtr result, UIntPtr index);

        [DllImport(Library, EntryPoint = "morflow_result_free", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ResultFree(IntPtr result);

        /// <summary>NUL-terminated UTF-8, as the C interface takes strings.</summary>
        internal static byte[] Utf8(string text)
        {
            var bytes = new byte[Encoding.UTF8.GetByteCount(text) + 1];
            Encoding.UTF8.GetBytes(text, 0, text.Length, bytes, 0);
            return bytes;
        }

        /// <summary>A NUL-terminated UTF-8 string from native memory (null for a null pointer).</summary>
        internal static string ReadUtf8(IntPtr text)
        {
            if (text == IntPtr.Zero) return null;
            var length = 0;
            while (Marshal.ReadByte(text, length) != 0) length++;
            var bytes = new byte[length];
            Marshal.Copy(text, bytes, 0, length);
            return Encoding.UTF8.GetString(bytes);
        }

        /// <summary>Reads and frees a string the engine allocated.</summary>
        internal static string TakeString(IntPtr text)
        {
            try
            {
                return ReadUtf8(text);
            }
            finally
            {
                if (text != IntPtr.Zero) StringFree(text);
            }
        }
    }
}
