import os
import wave
import numpy as np
import morflow

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load pipeline and decode input WAV
    pipeline = morflow.load("audio_split_pipeline.morf")
    with wave.open("input.wav", "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    int_samples = np.frombuffer(raw_data, dtype=np.int16)
    float_samples = (int_samples.astype(np.float32) / 32768.0).reshape(-1, n_channels).T
    float_samples = np.ascontiguousarray(float_samples)

    print(f"Loaded pipeline: {pipeline}")
    print(f"Input audio: {n_channels} channels, {framerate} Hz, {float_samples.shape[1]} samples")

    # 2. Execute Morflow multi-channel split pipeline (returns a dict of emitted outputs)
    outputs = pipeline.run(float_samples)
    print(f"Emitted outputs: {list(outputs.keys())}")

    # 3. Save each emitted channel to its own WAV file
    for name, channel_data in outputs.items():
        out_filename = f"{name}.wav"
        samples = np.clip(channel_data * 32767.0, -32768.0, 32767.0).astype(np.int16)

        with wave.open(out_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(samples.tobytes())

        print(f"Saved channel output '{name}' to {out_filename}")

    print("Multi-channel audio split completed successfully!")

if __name__ == "__main__":
    main()
