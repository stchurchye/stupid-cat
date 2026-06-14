import numpy as np
import pytest

from stupid_cat.config import PreprocessConfig
from stupid_cat.preprocess import preprocess_frame


def test_preprocess_bgr_preserves_shape() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    cfg = PreprocessConfig(input_mode="bgr", clahe_enabled=False)
    out = preprocess_frame(frame, cfg)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_preprocess_gray3_outputs_three_channels() -> None:
    frame = np.full((32, 40, 3), 128, dtype=np.uint8)
    cfg = PreprocessConfig(input_mode="gray3", clahe_enabled=False)
    out = preprocess_frame(frame, cfg)
    assert out.shape == (32, 40, 3)
    assert np.allclose(out[:, :, 0], out[:, :, 1])
    assert np.allclose(out[:, :, 1], out[:, :, 2])


def test_preprocess_rejects_invalid_input_mode() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    cfg = PreprocessConfig(input_mode="rgb")
    with pytest.raises(ValueError, match="unsupported input_mode"):
        preprocess_frame(frame, cfg)


def test_preprocess_clahe_preserves_shape_and_changes_pixels() -> None:
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[10:50, 10:70] = 40
    frame[20:40, 20:60] = 200
    cfg = PreprocessConfig(input_mode="bgr", clahe_enabled=True)
    out = preprocess_frame(frame, cfg)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
