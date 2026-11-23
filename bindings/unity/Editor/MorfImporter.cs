// Imports each .morf file in Assets as a MorflowPipelineAsset, which scripts
// reference from a field and compile with Morflow.FromAsset. Builds include
// the pipeline when a scene or prefab uses it, like any other asset.
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace MorflowUnity.Editor
{
    [ScriptedImporter(1, "morf")]
    internal sealed class MorfImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = MorflowPipelineAsset.Create(File.ReadAllText(ctx.assetPath));
            asset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            ctx.AddObjectToAsset("pipeline", asset);
            ctx.SetMainObject(asset);
        }
    }

    /// <summary>Shows a pipeline's parameters, or why it does not compile.</summary>
    [CustomEditor(typeof(MorflowPipelineAsset))]
    internal sealed class MorflowPipelineAssetEditor : UnityEditor.Editor
    {
        private static readonly Dictionary<string, string> Checked = new Dictionary<string, string>();

        public override void OnInspectorGUI()
        {
            var asset = (MorflowPipelineAsset)target;
            EditorGUILayout.HelpBox(Check(asset.Text), MessageType.None);
            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.TextArea(asset.Text, GUILayout.ExpandHeight(true));
            }
        }

        /// <summary>A summary of the pipeline, compiled once per text.</summary>
        private static string Check(string text)
        {
            if (Checked.TryGetValue(text, out var summary)) return summary;
            try
            {
                using (var pipeline = Morflow.FromStr(text))
                {
                    summary = pipeline.Params.Count == 0
                        ? "Compiles. It takes no parameters."
                        : $"Compiles. Parameters, in order: {string.Join(", ", pipeline.Params)}";
                }
            }
            catch (MorflowException e)
            {
                summary = e.Message.Contains("not bundled") || e.Message.Contains("No prepared")
                    ? $"{e.Message}\nRun morflow prep --prod-target unity on this file to list the packs it needs."
                    : e.Message;
            }
            Checked[text] = summary;
            return summary;
        }
    }
}
