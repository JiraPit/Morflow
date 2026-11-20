import org.morflow.Morflow;
import org.morflow.MorflowTensor;
import org.morflow.Pipeline;

import java.util.Arrays;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        // 1. Load pipeline and build a [4, 3] tensor: one row per time step, one column per sensor
        Pipeline pipeline = Morflow.load("tensor_loops_pipeline.morf");
        float[] readings = {
            -1.0f, 2.0f, 0.5f,
            3.0f, -1.0f, 1.5f,
            2.0f, 4.0f, -0.5f,
            0.5f, 1.0f, 12.0f,
        };

        System.out.println("Loaded pipeline: " + pipeline);
        System.out.println("Input readings: shape [4, 3]");

        // 2. Execute Morflow pipeline
        Map<String, MorflowTensor> outputs = pipeline.runToMap(MorflowTensor.fromFloatArray(readings, new int[] {4, 3}));

        // 3. Print each emitted tensor, one row per line
        for (String name : new String[] {"running_totals", "step_weights", "aligned", "checked"}) {
            MorflowTensor tensor = outputs.get(name);
            float[] values = tensor.toFloatArray();
            int[] shape = tensor.getShape();
            System.out.println(name + " " + Arrays.toString(shape) + ":");
            for (int row = 0; row < shape[0]; row++) {
                System.out.println("  " + Arrays.toString(Arrays.copyOfRange(values, row * shape[1], (row + 1) * shape[1])));
            }
        }
    }
}
