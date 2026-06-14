"""IR frame preprocessing (spec §7.2)."""

from __future__ import annotations

import cv2
import numpy as np

from stupid_cat.config import PreprocessConfig

_CLAHE: cv2.CLAHE | None = None


def preprocess_frame(frame_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply input_mode, optional CLAHE, optional denoise. Returns BGR uint8."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be HxWx3 BGR")

    if cfg.input_mode == "gray3":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        working = cv2.merge([gray, gray, gray])
    elif cfg.input_mode == "bgr":
        working = frame_bgr.copy()
    else:
        raise ValueError(f"unsupported input_mode: {cfg.input_mode}")

    if cfg.clahe_enabled:
        working = _apply_clahe(working)

    if cfg.denoise_enabled:
        working = cv2.fastNlMeansDenoisingColored(working, None, 3, 3, 7, 21)

    return working


def _get_clahe() -> cv2.CLAHE:
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def _apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _get_clahe().apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
