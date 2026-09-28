import os
import morflow

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load pipeline and read input audio file directly
    pipeline = morflow.load("audio_pipeline.morf")
    with open("input.wav", "rb") as f:
        input_bytes = f.read()

    print(f"Loaded pipeline: {pipeline}")
    print(f"Input audio binary: {len(input_bytes)} bytes")

    # 2. Execute Morflow DSP pipeline (handles WAV decoding, DSP chain, and WAV encoding)
    output_wav = pipeline.run(input_bytes)

    # 3. Save output WAV directly to disk
    with open("output.wav", "wb") as f:
        f.write(output_wav)

    print(f"Output audio: {len(output_wav)} bytes, saved directly to output.wav")
    print("Successfully processed audio with Python host!")

if __name__ == "__main__":
    main()
