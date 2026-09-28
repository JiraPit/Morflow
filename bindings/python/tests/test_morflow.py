import numpy as np
import pytest
import morflow

def test_identity_pipeline_f32():
    pipeline = morflow.from_str("""
        accept $data
        $data >> identity >> emit
    """)
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = pipeline.run(inp)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, inp)
    assert out.dtype == np.float32

def test_identity_pipeline_u8():
    pipeline = morflow.from_str("""
        accept $data
        $data >> identity >> emit
    """)
    inp = np.array([10, 20, 30, 40], dtype=np.uint8)
    out = pipeline.run(inp)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, inp)
    assert out.dtype == np.uint8

def test_multi_emit_pipeline():
    pipeline = morflow.from_str("""
        accept $data
        $data[0:2] >> identity >> emit("part_a")
        $data[2:4] >> identity >> emit("part_b")
    """)
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    outputs = pipeline.run(inp)
    assert isinstance(outputs, dict)
    assert "part_a" in outputs
    assert "part_b" in outputs
    assert np.allclose(outputs["part_a"], [1.0, 2.0])
    assert np.allclose(outputs["part_b"], [3.0, 4.0])

def test_syntax_error_handling():
    with pytest.raises(ValueError):
        morflow.from_str("accept $invalid >>> broken")

def test_file_not_found_handling():
    with pytest.raises(OSError):
        morflow.load("non_existent_file.morf")

def test_raw_bytes_payload():
    pipeline = morflow.from_str("""
        accept $data
        $data >> identity >> emit
    """)
    inp = b"hello morflow"
    out = pipeline.run(inp)
    assert out == inp

def test_audio_pipeline_raw_pcm_and_wav():
    pipeline = morflow.from_str("""
        import audio_essentials.latest
        accept $data
        $data >> to_audio(channels=2, sample_rate=44100, dtype="i16") >> gain(linear=2.0) >> to_wav >> emit
    """)
    # 2 channels, 2 samples (1.0 and -0.5 scaled in 16-bit)
    raw_pcm = np.array([16384, -8192, 16384, -8192], dtype=np.int16).tobytes()
    out = pipeline.run(raw_pcm)
    assert isinstance(out, bytes)
    assert len(out) == 44 + 8 # 44 byte header + 8 bytes data
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"

if __name__ == "__main__":
    pytest.main([__file__])
