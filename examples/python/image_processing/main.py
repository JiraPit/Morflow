import os
import sys
import numpy as np
from PIL import Image
import morflow

def main():
    # Ensure working directory is this script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load pipeline and decode input image
    pipeline = morflow.load("image_pipeline.morf")
    img = Image.open("input.png").convert("RGB")
    np_img = np.array(img, dtype=np.uint8)

    print(f"Loaded pipeline: {pipeline}")
    print(f"Input image shape: {np_img.shape}, dtype: {np_img.dtype}")

    # 2. Execute Morflow pipeline
    result = pipeline.run(morflow.Image(np_img, color="rgb"))

    # 3. Save output image
    print(f"Output image shape: {result.shape}, dtype: {result.dtype}")
    out_img = Image.fromarray(result)
    out_img.save("output.png")
    print("Successfully processed image with Python host and saved to output.png")

if __name__ == "__main__":
    main()
