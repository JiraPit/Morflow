using System;

namespace MorflowUnity
{
    /// <summary>
    /// A pipeline could not be compiled or run: a parse, compile, type or
    /// action error, with the engine's message.
    /// </summary>
    public sealed class MorflowException : Exception
    {
        public MorflowException(string message) : base(message)
        {
        }

        public MorflowException(string message, Exception inner) : base(message, inner)
        {
        }
    }
}
