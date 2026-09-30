package org.morflow;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * An executable Morflow pipeline instance.
 * <p>
 * Implements {@link AutoCloseable} to safely manage native engine resources.
 */
public class Pipeline implements AutoCloseable {
    static {
        NativeLoader.load();
    }

    private final long nativeHandle;
    private boolean closed = false;

    private Pipeline(long nativeHandle) {
        if (nativeHandle == 0) {
            throw new MorflowException("Failed to initialize native Morflow pipeline");
        }
        this.nativeHandle = nativeHandle;
    }

    /**
     * Compiles and loads a `.morf` pipeline file from disk.
     *
     * @param path File path to the `.morf` file.
     * @return Pipeline instance.
     */
    public static Pipeline load(String path) {
        long handle = nativeLoad(path);
        return new Pipeline(handle);
    }

    /**
     * Compiles a `.morf` pipeline DSL source string directly.
     *
     * @param source Raw DSL string.
     * @return Pipeline instance.
     */
    public static Pipeline fromStr(String source) {
        long handle = nativeFromStr(source);
        return new Pipeline(handle);
    }

    /**
     * Returns parameter names declared in the pipeline via {@code accept $param}.
     */
    public List<String> getParams() {
        checkClosed();
        String[] params = nativeGetParams(nativeHandle);
        return params != null ? Arrays.asList(params) : List.of();
    }

    /**
     * Executes the pipeline with a multi-dimensional {@link MorflowTensor} input.
     *
     * @param input Input tensor or buffer.
     * @return Single emitted output tensor.
     */
    public MorflowTensor run(MorflowTensor input) {
        checkClosed();
        return nativeRun(nativeHandle, input);
    }

    /**
     * Executes the pipeline with a raw binary byte array (e.g. WAV, PNG, or raw PCM bytes).
     *
     * @param rawBytes Raw byte array.
     * @return Single emitted output tensor.
     */
    public MorflowTensor run(byte[] rawBytes) {
        return run(MorflowTensor.fromBytes(rawBytes));
    }

    /**
     * Executes the pipeline with a direct {@link ByteBuffer}.
     *
     * @param directBuffer Direct byte buffer allocated via {@link ByteBuffer#allocateDirect(int)}.
     * @param shape Dimensions array (e.g. [H, W, C] or [channels, samples]).
     * @param dtype Data type ("f32", "u8", "i32", "raw").
     * @return Single emitted output tensor.
     */
    public MorflowTensor run(ByteBuffer directBuffer, int[] shape, String dtype) {
        return run(directBuffer, shape, dtype, null, null, null, 0, 0);
    }

    /**
     * Executes the pipeline with a direct {@link ByteBuffer} and an explicit payload type.
     *
     * <p>There is no type inference. With a null {@code payloadType} the input is
     * sent as a plain tensor, regardless of its shape.
     *
     * @param directBuffer Direct byte buffer allocated via {@link ByteBuffer#allocateDirect(int)}.
     * @param shape        Dimensions array (e.g. [H, W, C] or [channels, samples]).
     * @param dtype        Data type ("f32", "u8", "i32", "raw").
     * @param payloadType  "tensor", "image", or "audio". Null means a plain tensor.
     * @param colorSpace   Color space for "image" ("grayscale", "rgb", "rgba", "bgr", "bgra").
     * @param layout       "hwc"/"chw" for images, "planar"/"interleaved" for audio.
     * @param sampleRate   Sample rate in Hz for "audio". 0 defaults to 44100.
     * @param channels     Channel count for "audio". 0 defaults to the leading dimension.
     * @return Single emitted output tensor.
     */
    public MorflowTensor run(
            ByteBuffer directBuffer,
            int[] shape,
            String dtype,
            String payloadType,
            String colorSpace,
            String layout,
            int sampleRate,
            int channels) {
        checkClosed();
        return nativeRunDirect(
                nativeHandle, directBuffer, shape, dtype, payloadType, colorSpace, layout, sampleRate, channels);
    }

    /**
     * Executes the pipeline with a 32-bit floating point array and shape.
     *
     * @param floatArray Float array data.
     * @param shape Dimensions array.
     * @return Single emitted output tensor.
     */
    public MorflowTensor run(float[] floatArray, int[] shape) {
        return run(MorflowTensor.fromFloatArray(floatArray, shape));
    }

    /**
     * Executes the pipeline and returns a dictionary of all named emitted streams.
     *
     * @param input Input tensor.
     * @return Map of stream name to output tensor.
     */
    public Map<String, MorflowTensor> runAll(MorflowTensor input) {
        checkClosed();
        return nativeRunAll(nativeHandle, input);
    }

    /**
     * Executes the pipeline with positional parameters bound in declaration
     * order. Each value is a {@link MorflowTensor}, a {@code byte[]} (raw
     * bytes), a number ({@code Scalar} or {@code *Arg}), a {@link String}
     * ({@code StrArg}), or a {@link Boolean} ({@code BoolArg}). Parameters
     * without a supplied value use their declared default.
     *
     * @param args Positional parameter values.
     * @return Single emitted output tensor.
     */
    public MorflowTensor runArgs(Object... args) {
        checkClosed();
        return nativeRunArgs(nativeHandle, args);
    }

    /**
     * Executes the pipeline with positional parameters bound in declaration
     * order and returns a dictionary of all named emitted streams.
     *
     * @param args Positional parameter values.
     * @return Map of stream name to output tensor.
     */
    public Map<String, MorflowTensor> runAllArgs(Object... args) {
        checkClosed();
        return nativeRunAllArgs(nativeHandle, args);
    }

    /**
     * Executes the pipeline with raw binary bytes and returns all named emitted streams.
     *
     * @param rawBytes Raw byte array.
     * @return Map of stream name to output tensor.
     */
    public Map<String, MorflowTensor> runAll(byte[] rawBytes) {
        return runAll(MorflowTensor.fromBytes(rawBytes));
    }

    /**
     * Releases native pipeline resources.
     */
    @Override
    public synchronized void close() {
        if (!closed) {
            nativeDestroy(nativeHandle);
            closed = true;
        }
    }

    private void checkClosed() {
        if (closed) {
            throw new IllegalStateException("Morflow Pipeline has already been closed");
        }
    }

    @Override
    public String toString() {
        return "<Morflow Pipeline params=" + (closed ? "closed" : getParams()) + ">";
    }

    // -----------------------------------------------------------------------
    // Native JNI method declarations
    // -----------------------------------------------------------------------
    private static native long nativeLoad(String path);
    private static native long nativeFromStr(String source);
    private static native String[] nativeGetParams(long handle);
    private static native MorflowTensor nativeRun(long handle, MorflowTensor input);
    private static native MorflowTensor nativeRunDirect(
            long handle,
            ByteBuffer buffer,
            int[] shape,
            String dtype,
            String payloadType,
            String colorSpace,
            String layout,
            int sampleRate,
            int channels);
    private static native Map<String, MorflowTensor> nativeRunAll(long handle, MorflowTensor input);
    private static native MorflowTensor nativeRunArgs(long handle, Object[] args);
    private static native Map<String, MorflowTensor> nativeRunAllArgs(long handle, Object[] args);
    private static native void nativeDestroy(long handle);
}
