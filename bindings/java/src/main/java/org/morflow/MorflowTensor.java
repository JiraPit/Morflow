package org.morflow;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

/**
 * A multi-dimensional tensor and binary buffer representation in Java with zero-copy buffer accessors.
 */
public class MorflowTensor {
    private final ByteBuffer data;
    private final int[] shape;
    private final String dtype;
    private final String payloadType;
    private final String colorSpace;
    private final String layout;
    private final int sampleRate;
    private final int channels;

    /**
     * Constructs a MorflowTensor wrapping a native or direct {@link ByteBuffer}.
     *
     * <p>The payload is sent as a plain tensor. Use {@link #asImage} or
     * {@link #asAudio} to send it as an image or audio payload instead; there is
     * no type inference based on the shape.
     *
     * @param data Native direct ByteBuffer.
     * @param shape Dimensions array (e.g. [H, W, C] or [channels, samples]).
     * @param dtype Data type ("f32", "u8", "i32", "raw").
     */
    public MorflowTensor(ByteBuffer data, int[] shape, String dtype) {
        this.data = data.order(ByteOrder.LITTLE_ENDIAN);
        this.shape = shape != null ? shape : new int[]{data.capacity()};
        this.dtype = dtype != null ? dtype : "raw";
        this.payloadType = null;
        this.colorSpace = null;
        this.layout = null;
        this.sampleRate = 0;
        this.channels = 0;
    }

    private MorflowTensor(
            ByteBuffer data,
            int[] shape,
            String dtype,
            String payloadType,
            String colorSpace,
            String layout,
            int sampleRate,
            int channels) {
        this.data = data.order(ByteOrder.LITTLE_ENDIAN);
        this.shape = shape != null ? shape : new int[]{data.capacity()};
        this.dtype = dtype != null ? dtype : "raw";
        this.payloadType = payloadType;
        this.colorSpace = colorSpace;
        this.layout = layout;
        this.sampleRate = sampleRate;
        this.channels = channels;
    }

    /**
     * Returns a copy of this tensor that is sent to the engine as an image payload.
     *
     * @param colorSpace Color space ("grayscale", "rgb", "rgba", "bgr", "bgra").
     *                    Inferred from the trailing dimension when null.
     * @param layout Memory layout ("hwc" or "chw"). Defaults to "hwc".
     */
    public MorflowTensor asImage(String colorSpace, String layout) {
        return new MorflowTensor(data, shape, dtype, "image", colorSpace, layout, 0, 0);
    }

    /**
     * Returns a copy of this tensor that is sent to the engine as an image payload
     * using the default "hwc" layout.
     */
    public MorflowTensor asImage(String colorSpace) {
        return asImage(colorSpace, null);
    }

    /**
     * Returns a copy of this tensor that is sent to the engine as an audio payload.
     *
     * @param sampleRate Sample rate in Hz. Defaults to 44100 when &lt;= 0.
     * @param channels    Channel count. Defaults to the leading dimension when &lt;= 0.
     * @param layout      Memory layout ("planar" or "interleaved"). Defaults to "planar".
     */
    public MorflowTensor asAudio(int sampleRate, int channels, String layout) {
        return new MorflowTensor(data, shape, dtype, "audio", null, layout, sampleRate, channels);
    }

    /**
     * Returns a copy of this tensor that is sent to the engine as an audio payload
     * with the default planar layout and a 44100 Hz sample rate.
     */
    public MorflowTensor asAudio() {
        return asAudio(0, 0, null);
    }

    /**
     * Returns the requested payload type ("tensor", "image", "audio"), or null for
     * a plain tensor.
     */
    public String getPayloadType() {
        return payloadType;
    }

    /**
     * Returns the requested color space for an image payload, or null.
     */
    public String getColorSpace() {
        return colorSpace;
    }

    /**
     * Returns the requested memory layout for an image or audio payload, or null.
     */
    public String getLayout() {
        return layout;
    }

    /**
     * Returns the requested sample rate for an audio payload, or 0 when unset.
     */
    public int getSampleRate() {
        return sampleRate;
    }

    /**
     * Returns the requested channel count for an audio payload, or 0 when unset.
     */
    public int getChannels() {
        return channels;
    }

    /**
     * Returns the array of dimensions.
     */
    public int[] getShape() {
        return shape;
    }

    /**
     * Returns the number of dimensions (rank).
     */
    public int getRank() {
        return shape.length;
    }

    /**
     * Returns the data type string ("f32", "u8", "i32", "raw").
     */
    public String getDtype() {
        return dtype;
    }

    /**
     * Returns the underlying direct {@link ByteBuffer}.
     */
    public ByteBuffer getData() {
        return data;
    }

    /**
     * Returns the total number of elements in the tensor.
     */
    public int getElementCount() {
        if (shape.length == 0) return 0;
        int count = 1;
        for (int dim : shape) {
            count *= dim;
        }
        return count;
    }

    /**
     * Copies and returns the underlying data as a byte array.
     */
    public byte[] toByteArray() {
        ByteBuffer buf = data.duplicate();
        buf.rewind();
        byte[] arr = new byte[buf.remaining()];
        buf.get(arr);
        return arr;
    }

    /**
     * Copies and returns the underlying data as a 32-bit floating point array.
     */
    public float[] toFloatArray() {
        ByteBuffer buf = data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        buf.rewind();
        FloatBuffer fb = buf.asFloatBuffer();
        float[] arr = new float[fb.remaining()];
        fb.get(arr);
        return arr;
    }

    /**
     * Copies and returns the underlying data as a 32-bit integer array.
     */
    public int[] toIntArray() {
        ByteBuffer buf = data.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        buf.rewind();
        IntBuffer ib = buf.asIntBuffer();
        int[] arr = new int[ib.remaining()];
        ib.get(arr);
        return arr;
    }

    /**
     * Creates a 1D tensor wrapping raw binary bytes (e.g. for audio WAV or image files).
     */
    public static MorflowTensor fromBytes(byte[] bytes) {
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.put(bytes);
        buf.flip();
        return new MorflowTensor(buf, new int[]{bytes.length}, "raw");
    }

    /**
     * Creates a tensor from an unsigned 8-bit byte array and shape.
     */
    public static MorflowTensor fromByteArray(byte[] data, int[] shape) {
        ByteBuffer buf = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.put(data);
        buf.flip();
        return new MorflowTensor(buf, shape, "u8");
    }

    /**
     * Creates a tensor from a 32-bit floating point array and shape.
     */
    public static MorflowTensor fromFloatArray(float[] data, int[] shape) {
        ByteBuffer buf = ByteBuffer.allocateDirect(data.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(data);
        return new MorflowTensor(buf, shape, "f32");
    }

    /**
     * Creates a tensor from a direct ByteBuffer.
     */
    public static MorflowTensor fromDirectBuffer(ByteBuffer directBuf, int[] shape, String dtype) {
        if (!directBuf.isDirect()) {
            throw new IllegalArgumentException("ByteBuffer must be direct (allocated via ByteBuffer.allocateDirect)");
        }
        return new MorflowTensor(directBuf, shape, dtype);
    }

    @Override
    public String toString() {
        return "MorflowTensor{shape=" + Arrays.toString(shape) + ", dtype='" + dtype + "', bytes=" + data.capacity() + "}";
    }
}
