# `to_wav`

Encodes structured `Audio` or multidimensional sample `Tensor` directly into a complete, standard RIFF/WAVE file byte buffer (`Payload::Data`), ready to be written directly to a `.wav` file on disk.

## Interface
- **Input Type**: `DataType::Audio` (`Payload::Audio`)
- **Output Type**: `DataType::Bytes` (`Payload::Data`)
- **Supported DTypes**: `F32` (normalized `[-1.0, 1.0]`)
- **Supported Layouts**: `Planar [Channels, Samples]` or `Interleaved [Samples, Channels]`

## Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` / `format` / `bits` | `string` | `"i16"` | Target audio encoding in the WAV header: `"i16"` (16-bit PCM), `"f32"` (32-bit float), `"i24"` (24-bit PCM), `"i32"` (32-bit PCM), `"u8"` (8-bit PCM). |
| `sample_rate` | `int` | *Audio sample rate* or `44100` | Optional sample rate override written to the WAV header. |
| `clip` | `bool` | `true` | Clamp float samples to `[-1.0, 1.0]` before quantization to prevent integer overflow. |

## Behavior
- Interleaves multichannel planar audio frames and formats sample data into the standard RIFF/WAVE container.
- Generates a valid 44-byte RIFF header (`RIFF`, `WAVE`, `fmt `, `data`) with proper `AudioFormat` (1 for integer PCM, 3 for IEEE Float), `ByteRate`, `BlockAlign`, and data chunk sizes.
- Emits binary `Payload::Data` which can be directly saved to a `.wav` file by host environments (Python `open("out.wav", "wb").write(bytes)`, JS `fs.writeFileSync("out.wav", buf)`, Rust `fs::write("out.wav", buf)`).
- Parallelized frame interleaving and sample quantization using Rayon.
