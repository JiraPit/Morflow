// The project's Morflow settings, ProjectSettings/Morflow.json, written by
// `morflow prep --prod-target unity`:
//
//   {
//     "packs": { "audio_basics": "0.2.5" },
//     "releaseUrl": "file:///path/to/staged-releases"   (optional)
//   }
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace MorflowUnity.Editor
{
    internal sealed class Settings
    {
        internal const string File = "ProjectSettings/Morflow.json";

        private static readonly Regex PackName = new Regex("^[a-z0-9_]+$");
        private static readonly Regex Version = new Regex(@"^\d+\.\d+\.\d+$");

        /// <summary>The packs to bundle: name -> exact version, sorted by name.</summary>
        public SortedDictionary<string, string> Packs { get; } = new SortedDictionary<string, string>(StringComparer.Ordinal);

        /// <summary>Where releases are: -morflowReleaseUrl, MORFLOW_RELEASE_URL, releaseUrl, or null for GitHub. A URL or a folder.</summary>
        public string ReleaseUrl { get; private set; }

        /// <summary>The settings of the project in <paramref name="projectRoot"/>; none when the file is missing.</summary>
        public static Settings Read(string projectRoot)
        {
            var path = Path.Combine(projectRoot, File);
            var settings = System.IO.File.Exists(path) ? Parse(System.IO.File.ReadAllText(path)) : new Settings();
            var env = Environment.GetEnvironmentVariable("MORFLOW_RELEASE_URL");
            if (!string.IsNullOrEmpty(env)) settings.ReleaseUrl = env;
            // For CI that starts Unity in a container, where variables do not reach it.
            var args = Environment.GetCommandLineArgs();
            var at = Array.IndexOf(args, "-morflowReleaseUrl");
            if (at >= 0 && at + 1 < args.Length) settings.ReleaseUrl = args[at + 1];
            return settings;
        }

        public static Settings Parse(string text)
        {
            object root;
            try
            {
                root = new Json(text).Document();
            }
            catch (FormatException e)
            {
                throw new SettingsException($"Cannot read {File}: {e.Message}");
            }
            if (!(root is Dictionary<string, object> top)) throw new SettingsException($"{File} must hold a JSON object");
            var settings = new Settings();
            if (top.TryGetValue("packs", out var packs))
            {
                if (!(packs is Dictionary<string, object> list)) throw new SettingsException($"\"packs\" in {File} must be an object");
                foreach (var entry in list)
                {
                    if (!PackName.IsMatch(entry.Key)) throw new SettingsException($"{File} has an invalid pack name: {entry.Key}");
                    if (!(entry.Value is string version) || !Version.IsMatch(version))
                    {
                        throw new SettingsException($"packs.{entry.Key} in {File} must be an exact version such as \"0.3.4\"");
                    }
                    settings.Packs[entry.Key] = version;
                }
            }
            if (top.TryGetValue("releaseUrl", out var url))
            {
                if (!(url is string text2)) throw new SettingsException($"\"releaseUrl\" in {File} must be a string");
                settings.ReleaseUrl = text2;
            }
            return settings;
        }

        /// <summary>A short JSON reader: objects, arrays, strings, numbers, true, false, null.</summary>
        private sealed class Json
        {
            private readonly string text;
            private int at;

            public Json(string text) => this.text = text;

            public object Document()
            {
                var value = Value();
                Space();
                if (at != text.Length) throw Error("unexpected text after the JSON value");
                return value;
            }

            private object Value()
            {
                Space();
                if (at >= text.Length) throw Error("unexpected end");
                switch (text[at])
                {
                    case '{': return Object();
                    case '[': return Array();
                    case '"': return String();
                    case 't': return Word("true", true);
                    case 'f': return Word("false", false);
                    case 'n': return Word("null", null);
                    default: return Number();
                }
            }

            private Dictionary<string, object> Object()
            {
                var result = new Dictionary<string, object>();
                at++;
                Space();
                if (Peek('}')) return result;
                while (true)
                {
                    Space();
                    if (at >= text.Length || text[at] != '"') throw Error("expected a key");
                    var key = String();
                    Space();
                    Expect(':');
                    result[key] = Value();
                    Space();
                    if (Peek('}')) return result;
                    Expect(',');
                }
            }

            private List<object> Array()
            {
                var result = new List<object>();
                at++;
                Space();
                if (Peek(']')) return result;
                while (true)
                {
                    result.Add(Value());
                    Space();
                    if (Peek(']')) return result;
                    Expect(',');
                }
            }

            private string String()
            {
                var builder = new StringBuilder();
                at++;
                while (at < text.Length)
                {
                    var c = text[at++];
                    if (c == '"') return builder.ToString();
                    if (c != '\\')
                    {
                        builder.Append(c);
                        continue;
                    }
                    if (at >= text.Length) break;
                    var escape = text[at++];
                    switch (escape)
                    {
                        case 'n': builder.Append('\n'); break;
                        case 't': builder.Append('\t'); break;
                        case 'r': builder.Append('\r'); break;
                        case 'b': builder.Append('\b'); break;
                        case 'f': builder.Append('\f'); break;
                        case 'u':
                            if (at + 4 > text.Length) throw Error("bad \\u escape");
                            builder.Append((char)int.Parse(text.Substring(at, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                            at += 4;
                            break;
                        default: builder.Append(escape); break;
                    }
                }
                throw Error("unterminated string");
            }

            private object Number()
            {
                var start = at;
                while (at < text.Length && "+-0123456789.eE".IndexOf(text[at]) >= 0) at++;
                if (start == at) throw Error($"unexpected '{text[at]}'");
                return double.Parse(text.Substring(start, at - start), CultureInfo.InvariantCulture);
            }

            private object Word(string word, object value)
            {
                if (string.CompareOrdinal(text, at, word, 0, word.Length) != 0) throw Error($"unexpected '{text[at]}'");
                at += word.Length;
                return value;
            }

            private void Space()
            {
                while (at < text.Length && char.IsWhiteSpace(text[at])) at++;
            }

            private bool Peek(char c)
            {
                if (at < text.Length && text[at] == c)
                {
                    at++;
                    return true;
                }
                return false;
            }

            private void Expect(char c)
            {
                if (!Peek(c)) throw Error($"expected '{c}'");
            }

            private FormatException Error(string message)
            {
                var line = 1;
                for (var i = 0; i < Math.Min(at, text.Length); i++)
                {
                    if (text[i] == '\n') line++;
                }
                return new FormatException($"{message} on line {line}");
            }
        }
    }

    internal sealed class SettingsException : Exception
    {
        public SettingsException(string message) : base(message)
        {
        }
    }
}
