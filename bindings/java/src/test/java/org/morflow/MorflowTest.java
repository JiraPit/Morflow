package org.morflow;

import org.junit.jupiter.api.Test;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class MorflowTest {

    @Test
    public void testFromStrAndParams() {
        String dsl = """
            import base/latest
            accept Audio $audio_in
            accept IntArg $rate = 44100
            
            $audio_in >> base/latest/identity >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            List<String> params = pipeline.getParams();
            assertEquals(2, params.size());
            assertEquals("audio_in", params.get(0));
            assertEquals("rate", params.get(1));
        }
    }

    @Test
    public void testExecutionWithFloatArray() {
        String dsl = """
            accept Tensor $tensor
            $tensor >> base/latest/identity >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            float[] input = new float[]{1.0f, 2.5f, -3.0f, 4.25f};
            MorflowTensor output = pipeline.run(input, new int[]{4});

            assertNotNull(output);
            assertEquals("f32", output.getDtype());
            assertArrayEquals(new int[]{4}, output.getShape());
            assertEquals(4, output.getElementCount());

            float[] outArr = output.toFloatArray();
            assertArrayEquals(input, outArr, 1e-5f);
        }
    }

    @Test
    public void testExecutionWithDirectByteBuffer() {
        String dsl = """
            accept Tensor $tensor
            $tensor >> base/latest/identity >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            ByteBuffer buf = ByteBuffer.allocateDirect(12).order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(new float[]{10.0f, 20.0f, 30.0f});

            MorflowTensor output = pipeline.run(buf, new int[]{3}, "f32");
            assertNotNull(output);
            float[] outArr = output.toFloatArray();
            assertArrayEquals(new float[]{10.0f, 20.0f, 30.0f}, outArr, 1e-5f);
        }
    }

    @Test
    public void testMultipleNamedOutputs() {
        String dsl = """
            accept Tensor $audio
            $audio[0:2] >> base/latest/identity >> emit("low")
            $audio[2:4] >> base/latest/identity >> emit("high")
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            float[] input = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
            Map<String, MorflowTensor> outputs = pipeline.runAll(MorflowTensor.fromFloatArray(input, new int[]{4}));

            assertNotNull(outputs.get("low"));
            assertNotNull(outputs.get("high"));
            assertArrayEquals(new float[]{1.0f, 2.0f}, outputs.get("low").toFloatArray(), 1e-5f);
            assertArrayEquals(new float[]{3.0f, 4.0f}, outputs.get("high").toFloatArray(), 1e-5f);
        }
    }

    @Test
    public void testAudioToAudioAndToWavPipeline() {
        String dsl = """
            import audio_essentials/latest
            accept Bytes $data
            $data >> to_audio(channels=2, sample_rate=44100, dtype="i16") >> gain(linear=2.0) >> to_wav >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            ByteBuffer pcmBuf = ByteBuffer.allocateDirect(8).order(ByteOrder.LITTLE_ENDIAN);
            pcmBuf.putShort((short) 16384);
            pcmBuf.putShort((short) -8192);
            pcmBuf.putShort((short) 16384);
            pcmBuf.putShort((short) -8192);

            byte[] pcmBytes = new byte[8];
            pcmBuf.rewind();
            pcmBuf.get(pcmBytes);

            MorflowTensor output = pipeline.run(pcmBytes);
            byte[] wavBytes = output.toByteArray();

            assertEquals(44 + 8, wavBytes.length);
            assertEquals("RIFF", new String(wavBytes, 0, 4));
            assertEquals("WAVE", new String(wavBytes, 8, 4));
        }
    }

    @Test
    public void testBareArrayIsAPlainTensor() {
        // There is no type inference: a bare rank-2 [4, N] float32 array is a
        // plain tensor and reaches tensor actions without any wrapping.
        String dsl = """
            import tensor_essentials/latest
            accept Tensor $data
            $data >> reshape(shape="2, 1000") >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            float[] input = new float[2000];
            for (int i = 0; i < input.length; i++) input[i] = i;
            MorflowTensor output = pipeline.run(input, new int[]{4, 500});

            assertArrayEquals(new int[]{2, 1000}, output.getShape());
            assertArrayEquals(input, output.toFloatArray(), 0.0f);
        }
    }

    @Test
    public void testRank3ArrayIsNotInferredAsAnImage() {
        // A rank-3 array is no longer auto-promoted to an image. Image actions
        // declare DataType::Tensor input, so a bare array still runs without
        // any wrapping.
        String dsl = """
            from image_essentials/latest import to_image
            accept Tensor $image
            $image >> to_image >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            byte[] rgb = new byte[8 * 8 * 3];
            MorflowTensor bare = MorflowTensor.fromByteArray(rgb, new int[]{8, 8, 3});
            assertNull(bare.getPayloadType());

            MorflowTensor fromBare = pipeline.run(bare);
            assertArrayEquals(new int[]{8, 8, 3}, fromBare.getShape());
            assertEquals("u8", fromBare.getDtype());
        }
    }

    @Test
    public void testExplicitImagePayload() {
        // An asImage payload requires a pipeline that declares Image.
        String dsl = """
            from base/latest import to_tensor
            from image_essentials/latest import to_image
            accept Image $image
            $image >> to_tensor >> to_image >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            byte[] rgb = new byte[8 * 8 * 1];
            MorflowTensor fromImage = pipeline.run(
                    MorflowTensor.fromByteArray(rgb, new int[]{8, 8, 1}).asImage("grayscale"));
            assertArrayEquals(new int[]{8, 8}, fromImage.getShape());
        }
    }

    @Test
    public void testExplicitAudioPayload() {
        String dsl = """
            import audio_essentials/latest
            accept Audio $audio
            $audio >> to_wav >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            float[] samples = new float[2000];
            MorflowTensor input = MorflowTensor.fromFloatArray(samples, new int[]{2, 1000});

            assertThrows(MorflowException.class, () -> pipeline.run(input));

            MorflowTensor output = pipeline.run(input.asAudio(48000, 2, null));
            byte[] wavBytes = output.toByteArray();
            assertEquals("RIFF", new String(wavBytes, 0, 4));
            assertEquals("WAVE", new String(wavBytes, 8, 4));
        }
    }

    @Test
    public void testUnknownPayloadTypeRejected() {
        String dsl = """
            accept Tensor $data
            $data >> base/latest/identity >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            ByteBuffer buf = ByteBuffer.allocateDirect(8).order(ByteOrder.LITTLE_ENDIAN);
            MorflowException err = assertThrows(MorflowException.class, () -> {
                pipeline.run(buf, new int[]{2}, "f32", "hologram", null, null, 0, 0);
            });
            assertTrue(err.getMessage().contains("Unknown payloadType"),
                    "expected an unknown-payloadType error, got: " + err.getMessage());
        }
    }

    @Test
    public void testSyntaxErrorThrowsMorflowException() {
        assertThrows(MorflowException.class, () -> {
            Morflow.fromStr("accept Tensor $invalid >>> broken");
        });
    }

    @Test
    public void testFileNotFoundThrowsMorflowException() {
        assertThrows(MorflowException.class, () -> {
            Morflow.load("non_existent_file.morf");
        });
    }

    @Test
    public void testPipelineClose() {
        Pipeline pipeline = Morflow.fromStr("accept Tensor $a \n $a >> base/latest/identity >> emit");
        pipeline.close();
        assertThrows(IllegalStateException.class, () -> {
            pipeline.run(new byte[]{1, 2, 3});
        });
    }

    @Test
    public void testScalarParamAcceptsPlainNumber() {
        String dsl = """
            import nn_essentials/latest
            accept Scalar $value
            $value >> relu >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            MorflowTensor output = pipeline.runArgs(-3.0);
            assertEquals(0, output.getShape().length);
            assertArrayEquals(new float[]{0.0f}, output.toFloatArray(), 1e-5f);
        }
    }

    @Test
    public void testArgParamsPositionalWithDefault() {
        String dsl = """
            import audio_essentials/latest
            accept Bytes $data
            accept IntArg $rate = 48000
            $data >> to_audio(channels=2, sample_rate=$rate, dtype="i16") >> to_wav >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            byte[] pcm = new byte[8];

            byte[] viaDefault = pipeline.runArgs(pcm).toByteArray();
            assertEquals("RIFF", new String(viaDefault, 0, 4));

            byte[] viaArg = pipeline.runArgs(pcm, 22050).toByteArray();
            assertEquals("RIFF", new String(viaArg, 0, 4));
        }
    }

    @Test
    public void testMixedArgParams() {
        String dsl = """
            import audio_essentials/latest
            accept Audio $audio
            accept FloatArg $linear = 1.0
            accept BoolArg $routed = true
            accept Scalar $mix = 0.0
            $audio >> to_wav >> emit
        """;
        try (Pipeline pipeline = Morflow.fromStr(dsl)) {
            float[] samples = new float[128];
            MorflowTensor input = MorflowTensor.fromFloatArray(samples, new int[]{2, 64});

            Map<String, MorflowTensor> outputs = pipeline.runAllArgs(
                    input.asAudio(48000, 2, null), 2.0f, Boolean.FALSE, 0.5);
            assertNotNull(outputs);
        }
    }
    @Test
    public void testExactVersionAndRequiredImports() {
        String source = "from base/0.2.0 import identity\naccept Tensor $data\n$data >> identity >> emit";
        try (Pipeline pipeline = Morflow.fromStr(source)) {
            assertArrayEquals(new float[]{1.0f, 2.0f}, pipeline.run(new float[]{1.0f, 2.0f}, new int[]{2}).toFloatArray());
        }
        assertThrows(MorflowException.class, () -> Morflow.fromStr("accept Tensor $data\n$data >> identity >> emit"));
    }
}