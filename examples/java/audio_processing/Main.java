import org.morflow.Morflow;
import org.morflow.MorflowTensor;
import org.morflow.Pipeline;

import java.nio.file.Files;
import java.nio.file.Path;

public class Main {
    public static void main(String[] args) throws Exception {
        // 1. Load pipeline and read input WAV binary
        Pipeline pipeline = Morflow.load("audio_pipeline.morf");
        byte[] inputBytes = Files.readAllBytes(Path.of("input.wav"));

        System.out.println("Loaded pipeline: " + pipeline);
        System.out.println("Input audio binary: " + inputBytes.length + " bytes");

        // 2. Execute Morflow DSP pipeline (handles WAV decoding, DSP chain, and WAV encoding)
        MorflowTensor outputTensor = pipeline.run(inputBytes);
        byte[] outputWavBytes = outputTensor.toByteArray();

        // 3. Save output WAV directly to disk
        Files.write(Path.of("output.wav"), outputWavBytes);

        System.out.println("Output audio: " + outputWavBytes.length + " bytes, saved to output.wav");
        System.out.println("Successfully processed audio with Java host!");
    }
}
