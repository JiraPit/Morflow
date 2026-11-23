using UnityEngine;

namespace MorflowUnity
{
    /// <summary>
    /// A <c>.morf</c> pipeline file in the project's Assets. Give a script a
    /// field of this type, drag the file onto it, and compile it with
    /// <see cref="Morflow.FromAsset"/>.
    /// </summary>
    public sealed class MorflowPipelineAsset : ScriptableObject
    {
        [SerializeField, TextArea(5, 30)]
        private string text = "";

        /// <summary>The pipeline's source text.</summary>
        public string Text => text;

        /// <summary>Creates an asset holding <paramref name="source"/>, for the importer and tests.</summary>
        public static MorflowPipelineAsset Create(string source)
        {
            var asset = CreateInstance<MorflowPipelineAsset>();
            asset.text = source ?? "";
            return asset;
        }
    }
}
