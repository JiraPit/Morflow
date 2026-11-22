// Small Mach-O edits for Apple builds, done in C# so they work on any
// computer, not only on a Mac with Xcode's tools:
//   - Universal: joins thin macOS libraries (arm64 and x86_64) into one
//     universal library, as `lipo -create` does;
//   - SetInstallName: changes a library's install name, as
//     `install_name_tool -id` does, for iOS frameworks.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MorflowUnity.Editor
{
    internal static class MachO
    {
        private const uint Magic64 = 0xfeedfacf;
        private const uint FatMagic = 0xcafebabe;
        private const uint LcIdDylib = 0xd;
        private const uint CpuTypeArm64 = 0x0100000c;
        private const uint CpuTypeX86_64 = 0x01000007;

        /// <summary>The CPU type of a thin 64-bit Mach-O file.</summary>
        internal static uint CpuType(byte[] binary)
        {
            if (binary.Length < 32 || ReadLE(binary, 0) != Magic64) throw new InvalidDataException("Not a 64-bit Mach-O file");
            return ReadLE(binary, 4);
        }

        /// <summary>A universal (fat) binary holding each thin binary in <paramref name="slices"/>.</summary>
        internal static byte[] Universal(IList<byte[]> slices)
        {
            var headerSize = 8 + 20 * slices.Count;
            var entries = new List<(uint cpu, uint sub, uint offset, uint size, int align)>();
            var offset = (long)headerSize;
            foreach (var slice in slices)
            {
                var cpu = CpuType(slice);
                // Apple's linker aligns arm64 slices to 16 KB pages and Intel ones to 4 KB.
                var align = cpu == CpuTypeArm64 ? 14 : 12;
                offset = (offset + (1L << align) - 1) & ~((1L << align) - 1);
                entries.Add((cpu, ReadLE(slice, 8), (uint)offset, (uint)slice.Length, align));
                offset += slice.Length;
            }
            var result = new byte[offset];
            WriteBE(result, 0, FatMagic);
            WriteBE(result, 4, (uint)slices.Count);
            for (var i = 0; i < slices.Count; i++)
            {
                var at = 8 + 20 * i;
                var entry = entries[i];
                WriteBE(result, at, entry.cpu);
                WriteBE(result, at + 4, entry.sub);
                WriteBE(result, at + 8, entry.offset);
                WriteBE(result, at + 12, entry.size);
                WriteBE(result, at + 16, (uint)entry.align);
                Buffer.BlockCopy(slices[i], 0, result, (int)entry.offset, slices[i].Length);
            }
            return result;
        }

        /// <summary>The architectures in a universal binary, for checks.</summary>
        internal static List<uint> UniversalCpuTypes(byte[] binary)
        {
            if (binary.Length < 8 || ReadBE(binary, 0) != FatMagic) throw new InvalidDataException("Not a universal binary");
            var types = new List<uint>();
            for (var i = 0; i < ReadBE(binary, 4); i++) types.Add(ReadBE(binary, 8 + 20 * i));
            return types;
        }

        internal static string Describe(uint cpu) => cpu == CpuTypeArm64 ? "arm64" : cpu == CpuTypeX86_64 ? "x86_64" : $"cpu {cpu:x}";

        /// <summary>The install name (LC_ID_DYLIB) of a thin library.</summary>
        internal static string InstallName(byte[] binary)
        {
            var (at, size) = FindIdDylib(binary);
            var nameAt = (int)ReadLE(binary, at + 8);
            var end = at + nameAt;
            while (end < at + size && binary[end] != 0) end++;
            return Encoding.UTF8.GetString(binary, at + nameAt, end - at - nameAt);
        }

        /// <summary>
        /// Sets a thin library's install name in place. The load command keeps
        /// its size, so the new name must fit where the old one was; release
        /// libraries carry their long build path there.
        /// </summary>
        internal static void SetInstallName(byte[] binary, string name)
        {
            var (at, size) = FindIdDylib(binary);
            var nameAt = (int)ReadLE(binary, at + 8);
            var bytes = Encoding.UTF8.GetBytes(name);
            var room = size - nameAt;
            if (bytes.Length + 1 > room)
            {
                throw new InvalidDataException($"The install name {name} does not fit in the library's {room}-byte name field");
            }
            Array.Clear(binary, at + nameAt, room);
            Buffer.BlockCopy(bytes, 0, binary, at + nameAt, bytes.Length);
        }

        private static (int at, int size) FindIdDylib(byte[] binary)
        {
            if (binary.Length < 32 || ReadLE(binary, 0) != Magic64) throw new InvalidDataException("Not a 64-bit Mach-O library");
            var count = ReadLE(binary, 16);
            var at = 32;
            for (var i = 0; i < count; i++)
            {
                var cmd = ReadLE(binary, at);
                var size = (int)ReadLE(binary, at + 4);
                if (cmd == LcIdDylib) return (at, size);
                at += size;
            }
            throw new InvalidDataException("The library has no install name (LC_ID_DYLIB)");
        }

        private static uint ReadLE(byte[] b, int at) => (uint)(b[at] | b[at + 1] << 8 | b[at + 2] << 16 | b[at + 3] << 24);
        private static uint ReadBE(byte[] b, int at) => (uint)(b[at] << 24 | b[at + 1] << 16 | b[at + 2] << 8 | b[at + 3]);

        private static void WriteBE(byte[] b, int at, uint value)
        {
            b[at] = (byte)(value >> 24);
            b[at + 1] = (byte)(value >> 16);
            b[at + 2] = (byte)(value >> 8);
            b[at + 3] = (byte)value;
        }
    }
}
