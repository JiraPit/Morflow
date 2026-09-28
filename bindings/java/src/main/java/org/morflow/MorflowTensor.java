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

    /**
     * Constructs a MorflowTensor wrapping a native or direct {@link ByteBuffer}.
     *
     * @param data Native direct ByteBuffer.
     * @param shape Dimensions array (e.g. [H, W, C] or [channels, samples]).
     * @param dtype Data type ("f32", "u8", "i32", "raw").
     */
    public MorflowTensor(ByteBuffer data, int[] shape, String dtype) {
        this.data = data.order(ByteOrder.LITTLE_ENDIAN);
        this.shape = shape != null ? shape : new int[]{data.capacity()};
        this.dtype = dtype != null ? dtype : "raw";
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
