using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MorflowUnity
{
    /// <summary>
    /// A compiled pipeline.
    /// </summary>
    /// <remarks>
    /// Each run returns the single emitted output: a <see cref="MorflowTensor"/>,
    /// or a list for a Composite. The <c>…ToMap</c> methods return every named
    /// output. The <c>…Async</c> methods run on a worker thread, so the game
    /// keeps rendering, and continue on Unity's main thread when awaited there.
    /// <para>
    /// A pipeline can run on many threads at once. It holds native memory,
    /// released by <see cref="Dispose"/> or once it is garbage collected. Runs
    /// already in progress finish normally after <see cref="Dispose"/>.
    /// </para>
    /// </remarks>
    public sealed class MorflowPipeline : IDisposable
    {
        private readonly object gate = new object();
        private IntPtr handle;
        private int activeRuns;
        private bool disposed;

        /// <summary>Parameter names declared with <c>accept $param</c>, in declaration order.</summary>
        public IReadOnlyList<string> Params { get; }

        internal MorflowPipeline(IntPtr handle)
        {
            this.handle = handle;
            var names = new List<string>();
            var count = (ulong)Native.ParamCount(handle);
            for (ulong i = 0; i < count; i++) names.Add(Native.TakeString(Native.ParamName(handle, (UIntPtr)i)));
            Params = names.AsReadOnly();
        }

        ~MorflowPipeline()
        {
            // Nothing can be running once the pipeline is unreachable.
            if (handle != IntPtr.Zero) Native.PipelineFree(handle);
        }

        /// <summary>Runs the pipeline on this thread and returns its output.</summary>
        public object Run(object input = null) => Execute(arena => Native.Run(Use(), Convert.EncodeInput(input, arena), 0), false);

        /// <summary>Runs the pipeline on this thread and returns its output as <typeparamref name="T"/>.</summary>
        public T Run<T>(object input = null) where T : class => Cast<T>(Run(input));

        /// <summary>Runs the pipeline on this thread and returns every named output.</summary>
        public IReadOnlyDictionary<string, object> RunToMap(object input = null) =>
            (Dictionary<string, object>)Execute(arena => Native.Run(Use(), Convert.EncodeInput(input, arena), 1), true);

        /// <summary>
        /// Runs the pipeline on this thread with positional parameters bound in
        /// declaration order. Parameters without a value use their declared
        /// default. Each argument is anything <see cref="Run(object)"/> accepts,
        /// a number (Scalar or <c>*Arg</c>), a string (<c>StrArg</c>) or a bool
        /// (<c>BoolArg</c>).
        /// </summary>
        public object RunArgs(params object[] args) => Execute(arena =>
        {
            var values = Convert.EncodeArgs(args, arena);
            return Native.RunArgs(Use(), values, (UIntPtr)(ulong)(args?.Length ?? 0), 0);
        }, false);

        /// <summary>Runs the pipeline on this thread with positional parameters and returns every named output.</summary>
        public IReadOnlyDictionary<string, object> RunToMapArgs(params object[] args) => (Dictionary<string, object>)Execute(arena =>
        {
            var values = Convert.EncodeArgs(args, arena);
            return Native.RunArgs(Use(), values, (UIntPtr)(ulong)(args?.Length ?? 0), 1);
        }, true);

        /// <summary>Runs the pipeline on a worker thread.</summary>
        public Task<object> RunAsync(object input = null) => Task.Run(() => Run(input));

        /// <summary>Runs the pipeline on a worker thread and returns its output as <typeparamref name="T"/>.</summary>
        public async Task<T> RunAsync<T>(object input = null) where T : class =>
            // Not back on the caller's thread: a caller that blocks on .Result would wait forever.
            Cast<T>(await RunAsync(input).ConfigureAwait(false));

        /// <summary>Runs the pipeline on a worker thread and returns every named output.</summary>
        public Task<IReadOnlyDictionary<string, object>> RunToMapAsync(object input = null) => Task.Run(() => RunToMap(input));

        /// <summary>Runs the pipeline on a worker thread with positional parameters.</summary>
        public Task<object> RunArgsAsync(params object[] args) => Task.Run(() => RunArgs(args));

        /// <summary>Runs the pipeline on a worker thread with positional parameters and returns every named output.</summary>
        public Task<IReadOnlyDictionary<string, object>> RunToMapArgsAsync(params object[] args) => Task.Run(() => RunToMapArgs(args));

        /// <summary>
        /// Releases the pipeline's native memory once runs in progress finish.
        /// Later runs throw <see cref="ObjectDisposedException"/>.
        /// </summary>
        public void Dispose()
        {
            lock (gate)
            {
                if (disposed) return;
                disposed = true;
                if (activeRuns == 0) Free();
            }
            GC.SuppressFinalize(this);
        }

        public override string ToString() => $"MorflowPipeline(params: [{string.Join(", ", Params)}])";

        private IntPtr Use() => handle;

        private object Execute(Func<Arena, IntPtr> call, bool all)
        {
            lock (gate)
            {
                if (disposed) throw new ObjectDisposedException(nameof(MorflowPipeline));
                activeRuns++;
            }
            try
            {
                IntPtr result;
                using (var arena = new Arena()) result = call(arena);
                return Convert.DecodeResult(result, all);
            }
            finally
            {
                lock (gate)
                {
                    activeRuns--;
                    if (disposed && activeRuns == 0) Free();
                }
            }
        }

        private void Free()
        {
            if (handle == IntPtr.Zero) return;
            Native.PipelineFree(handle);
            handle = IntPtr.Zero;
        }

        private static T Cast<T>(object output) where T : class
        {
            if (output is T typed) return typed;
            throw new InvalidCastException($"The pipeline emitted {Describe(output)}, not {typeof(T).Name}");
        }

        private static string Describe(object output) =>
            output is MorflowTensor tensor ? tensor.ToString() : output is IReadOnlyList<object> ? "a Composite (a list of outputs)" : output?.GetType().Name ?? "nothing";
    }
}
