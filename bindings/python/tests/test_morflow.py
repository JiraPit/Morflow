import numpy as np
import pytest
import morflow


def test_identity_pipeline_f32():
    pipeline = morflow.from_str("""
        accept Tensor $data
        $data >> base/latest/identity >> emit
    """)
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = pipeline.run(inp)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, inp)
    assert out.dtype == np.float32


def test_identity_pipeline_u8():
    pipeline = morflow.from_str("""
        accept Tensor $data
        $data >> base/latest/identity >> emit
    """)
    inp = np.array([10, 20, 30, 40], dtype=np.uint8)
    out = pipeline.run(inp)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, inp)
    assert out.dtype == np.uint8


def test_multi_emit_pipeline():
    pipeline = morflow.from_str("""
        accept Tensor $data
        $data[0:2] >> base/latest/identity >> emit("part_a")
        $data[2:4] >> base/latest/identity >> emit("part_b")
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
        morflow.from_str("accept Tensor $invalid >>> broken")


def test_file_not_found_handling():
    with pytest.raises(OSError):
        morflow.load("non_existent_file.morf")


def test_bytes_payload():
    pipeline = morflow.from_str("""
        accept Bytes $data
        $data >> base/latest/identity >> emit
    """)
    inp = b"hello morflow"
    out = pipeline.run(inp)
    assert out == inp


def test_audio_pipeline_raw_pcm_and_wav():
    pipeline = morflow.from_str("""
        import audio_basics/latest
        accept Bytes $data
        $data >> to_audio(channels=2, sample_rate=44100, dtype="i16") >> gain(linear=2.0) >> to_wav >> emit
    """)
    # 2 channels, 2 samples (1.0 and -0.5 scaled in 16-bit)
    raw_pcm = np.array([16384, -8192, 16384, -8192], dtype=np.int16).tobytes()
    out = pipeline.run(raw_pcm)
    assert isinstance(out, bytes)
    assert len(out) == 44 + 8  # 44 byte header + 8 bytes data
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"


def test_bare_rank2_array_is_a_plain_tensor():
    """There is no type inference: a bare [4, N] feature matrix is a tensor and
    reaches tensor actions without any wrapping."""
    pipeline = morflow.from_str("""
        import tensor_basics/latest
        accept Tensor $data
        $data >> reshape(shape="2, 1000") >> emit
    """)
    inp = np.arange(2000, dtype=np.float32).reshape(4, 500)
    out = pipeline.run(inp)
    assert out.shape == (2, 1000)
    assert np.array_equal(out, inp.reshape(2, 1000))


def test_tensor_wrapper_forces_plain_tensor():
    """morflow.Tensor is the explicit form of the same plain-tensor behavior."""
    pipeline = morflow.from_str("""
        import tensor_basics/latest
        accept Tensor $data
        $data >> reshape(shape="2, 1000") >> emit
    """)
    inp = np.arange(2000, dtype=np.float32).reshape(4, 500)
    out = pipeline.run(morflow.Tensor(inp))
    assert out.shape == (2, 1000)
    assert np.array_equal(out, inp.reshape(2, 1000))


def test_audio_wrapper_forces_audio_payload():
    pipeline = morflow.from_str("""
        import audio_basics/latest
        accept Audio $audio
        $audio >> to_wav >> emit
    """)
    inp = np.zeros((2, 1000), dtype=np.float32)
    out = pipeline.run(morflow.Audio(inp, sample_rate=48000))
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"


def test_image_wrapper_forces_image_payload():
    pipeline = morflow.from_str("""
        from base/latest import to_tensor
        from image_basics/latest import to_image
        accept Image $image
        $image >> to_tensor >> to_image >> emit
    """)
    out = pipeline.run(morflow.Image(np.zeros((8, 8, 1), dtype=np.float32), color="grayscale"))
    assert out.shape == (8, 8)


def test_image_wrapper_rejects_bad_color_space():
    pipeline = morflow.from_str("""
        from base/latest import to_tensor
        accept Image $image
        $image >> to_tensor >> emit
    """)
    with pytest.raises(ValueError, match="Unknown color space"):
        pipeline.run(morflow.Image(np.zeros((8, 8, 3), dtype=np.float32), color="cmyk"))


def test_wrapper_rejects_non_ndarray():
    pipeline = morflow.from_str("""
        accept Tensor $data
        $data >> base/latest/identity >> emit
    """)
    with pytest.raises(TypeError, match="expects a contiguous NumPy array"):
        pipeline.run(morflow.Tensor([1.0, 2.0, 3.0]))


def test_bare_rank2_array_is_not_inferred_as_audio():
    """A bare [2, N] array is a plain tensor, so an audio-native action rejects
    it. Wrap it in morflow.Audio to send an audio payload."""
    pipeline = morflow.from_str("""
        from audio_basics/latest import to_wav
        accept Audio $audio
        $audio >> to_wav >> emit
    """)
    inp = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(TypeError, match="expected Audio"):
        pipeline.run(inp)
    assert pipeline.run(morflow.Audio(inp, sample_rate=44100))[:4] == b"RIFF"


def test_bare_rank3_array_runs_as_a_plain_tensor():
    """A rank-3 array is not auto-promoted to an image. Image actions declare
    DataType::Tensor input, so they accept it directly and to_image re-wraps
    the result as an image."""
    pipeline = morflow.from_str("""
        from image_basics/latest import to_image
        accept Tensor $image
        $image >> to_image >> emit
    """)
    out = pipeline.run(np.zeros((8, 8, 3), dtype=np.uint8))
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_scalar_param_accepts_plain_number():
    pipeline = morflow.from_str("""
        accept Scalar $value
        $value >> base/latest/identity >> emit
    """)
    out = pipeline.run(0.5)
    assert np.allclose(out, 0.5)


def test_scalar_param_survives_action_chain():
    pipeline = morflow.from_str("""
        import nn_basics/latest
        accept Scalar $value
        $value >> relu >> emit
    """)
    out = pipeline.run(-3.0)
    assert np.allclose(out, 0.0)


def test_intarg_param_with_default():
    pipeline = morflow.from_str("""
        import audio_basics/latest
        accept Bytes $data
        accept IntArg $rate = 48000
        $data >> to_audio(channels=2, sample_rate=$rate, dtype="i16") >> to_wav >> emit
    """)
    raw_pcm = np.array([0, 0, 0, 0], dtype=np.int16).tobytes()
    default_out = pipeline.run(raw_pcm)
    assert default_out[:4] == b"RIFF"
    explicit_out = pipeline.run(raw_pcm, 22050)
    assert explicit_out[:4] == b"RIFF"


def test_arg_params_and_scalar_together():
    pipeline = morflow.from_str("""
        import audio_basics/latest
        accept Audio $audio
        accept FloatArg $linear = 1.0
        accept BoolArg $routed = true
        accept Scalar $noise = 0.0
        $audio >> gain(linear=$linear) >> to_wav >> emit
    """)
    inp = np.zeros((2, 100), dtype=np.float32)
    out = pipeline.run(morflow.Audio(inp, sample_rate=44100), 2.0, False, 0.5)
    assert out[:4] == b"RIFF"


def test_exact_version_and_required_imports():
    pipeline = morflow.from_str("from base/0.3.1 import identity\naccept Tensor $data\n$data >> identity >> emit")
    np.testing.assert_array_equal(pipeline.run(np.array([1.0, 2.0], dtype=np.float32)), [1.0, 2.0])
    with pytest.raises(RuntimeError, match="not declared by the imports"):
        morflow.from_str("accept Tensor $data\n$data >> identity >> emit")


if __name__ == "__main__":
    pytest.main([__file__])


def test_qr_composite_component_selection():
    pipeline = morflow.from_str('''
        import linalg_basics/latest
        accept Tensor[3,2] $matrix
        $matrix >> qr >> $parts
        $parts[0] >> emit("q")
        $parts[1] >> emit("r")
        $parts >> matmul >> emit("reconstructed")
    ''')
    matrix = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    out = pipeline.run(matrix)
    assert out['q'].shape == (3, 2)
    assert out['r'].shape == (2, 2)
    assert np.allclose(out['q'] @ out['r'], matrix, atol=1e-5)

    assert np.allclose(out["reconstructed"], matrix, atol=1e-5)
