"""Run shared OpenCV image actions from a Python host."""

from pathlib import Path

import morflow
import numpy as np


def main():
    pipeline = morflow.load(str(Path(__file__).with_name("pipeline.morf")))
    image = np.full((480, 640, 3), 0.5, dtype=np.float32)
    output = pipeline.run(image)
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")


if __name__ == "__main__":
    main()
