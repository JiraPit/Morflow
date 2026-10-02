"""Run a BLAS QR pipeline from a Python host."""

from pathlib import Path

import morflow
import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    qr = morflow.load(str(directory / "qr.morf"))
    matrix = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    print("QR pipeline:\n", qr.run(matrix))


if __name__ == "__main__":
    main()
