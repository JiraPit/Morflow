import os
import wave
import numpy as np
import morflow

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Load pipeline and decode input WAV
    pipeline = morflow.load("audio_pipeline.morf")
    with wave.open("input.wav", "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert 16-bit PCM to planar float32 array [channels, samples]
    int_samples = np.frombuffer(raw_data, dtype=np.int16)
    float_samples = (int_samples.astype(np.float32) / 32768.0).reshape(-1, n_channels).T
    float_samples = np.ascontiguousarray(float_samples)

    print(f"Loaded pipeline: {pipeline}")
    print(f"Input audio: {n_channels} channels, {framerate} Hz, {float_samples.shape[1]} samples")

    # 2. Execute Morflow DSP pipeline
    output = pipeline.run(float_samples)

    # 3. Save output WAV (48000 Hz, 16-bit PCM)
    out_channels = output.shape[0] if output.ndim > 1 else 1
    out_samples = output.reshape(out_channels, -1).T
    out_int16 = np.clip(out_samples * 32767.0, -32768.0, 32767.0).astype(np.int16)

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(out_channels)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(out_int16.tobytes())

    print(f"Output audio: {out_channels} channels, 48000 Hz, saved to output.wav")
    print("Successfully processed audio with Python host!")

if __name__ == "__main__":
    main()
