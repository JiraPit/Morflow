import os
import morflow
import numpy as np

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load pipeline and build a [4, 3] tensor: one row per time step, one column per sensor
    pipeline = morflow.load("tensor_loops_pipeline.morf")
    readings = np.array(
        [
            [-1.0, 2.0, 0.5],
            [3.0, -1.0, 1.5],
            [2.0, 4.0, -0.5],
            [0.5, 1.0, 12.0],
        ],
        dtype=np.float32,
    )

    print(f"Loaded pipeline: {pipeline}")
    print(f"Input readings: shape {list(readings.shape)}")

    # 2. Execute Morflow pipeline (returns a dict of the emitted tensors)
    outputs = pipeline.run(readings)

    # 3. Print each emitted tensor
    for name in ("running_totals", "step_weights", "aligned", "checked"):
        print(f"{name} {list(outputs[name].shape)}:\n{outputs[name]}")

if __name__ == "__main__":
    main()
