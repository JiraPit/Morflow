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
            import base.latest
            accept $audio_in
            accept $rate = 44100
            
            $audio_in >> identity >> emit
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
            accept $tensor
            $tensor >> identity >> emit
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
            accept $tensor
            $tensor >> identity >> emit
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
            accept $audio
            $audio[0:2] >> identity >> emit("low")
            $audio[2:4] >> identity >> emit("high")
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
            import audio_essentials.latest
            accept $data
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
    public void testSyntaxErrorThrowsMorflowException() {
        assertThrows(MorflowException.class, () -> {
            Morflow.fromStr("invalid >>> broken syntax");
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
        Pipeline pipeline = Morflow.fromStr("accept $a \n $a >> identity >> emit");
        pipeline.close();
        assertThrows(IllegalStateException.class, () -> {
            pipeline.run(new byte[]{1, 2, 3});
        });
    }
}
