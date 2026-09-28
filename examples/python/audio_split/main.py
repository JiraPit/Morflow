import os
import morflow

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load pipeline and read input WAV bytes directly
    pipeline = morflow.load("audio_split_pipeline.morf")
    with open("input.wav", "rb") as f:
        input_bytes = f.read()

    print(f"Loaded pipeline: {pipeline}")
    print(f"Input audio binary: {len(input_bytes)} bytes")

    # 2. Execute Morflow multi-channel split pipeline (returns a dict of emitted WAV binaries)
    outputs = pipeline.run(input_bytes)
    print(f"Emitted outputs: {list(outputs.keys())}")

    # 3. Save each emitted channel directly to its own WAV file
    for name, wav_bytes in outputs.items():
        out_filename = f"{name}.wav"
        with open(out_filename, "wb") as f:
            f.write(wav_bytes)
        print(f"Saved channel output '{name}' ({len(wav_bytes)} bytes) to {out_filename}")

    print("Multi-channel audio split completed successfully!")

if __name__ == "__main__":
    main()
