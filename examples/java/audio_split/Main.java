import org.morflow.Morflow;
import org.morflow.MorflowTensor;
import org.morflow.Pipeline;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        // 1. Load pipeline and read input WAV binary
        Pipeline pipeline = Morflow.load("audio_split_pipeline.morf");
        byte[] inputBytes = Files.readAllBytes(Path.of("input.wav"));

        System.out.println("Loaded pipeline: " + pipeline);
        System.out.println("Input audio binary: " + inputBytes.length + " bytes");

        // 2. Execute Morflow multi-channel split pipeline
        Map<String, MorflowTensor> outputs = pipeline.runAll(inputBytes);
        System.out.println("Emitted outputs: " + String.join(", ", outputs.keySet()));

        // 3. Save each emitted channel directly to its own WAV file
        for (Map.Entry<String, MorflowTensor> entry : outputs.entrySet()) {
            String name = entry.getKey();
            byte[] wavBytes = entry.getValue().toByteArray();
            String outFilename = name + ".wav";
            Files.write(Path.of(outFilename), wavBytes);
            System.out.println("Saved channel output '" + name + "' (" + wavBytes.length + " bytes) to " + outFilename);
        }

        System.out.println("Multi-channel audio split completed successfully!");
    }
}
