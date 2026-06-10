"""YOLO cat detection and per-camera frame confirmation (spec §4.1, §8.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stupid_cat.config import InferenceConfig

BBox = tuple[float, float, float, float]
COCO_CAT_CLASS_ID = 15


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, max(x1, x2) - min(x1, x2)) * max(0.0, max(y1, y2) - min(y1, y2))


def select_primary_bbox(boxes: list[BBox]) -> BBox | None:
    """Return the largest-area bbox, or None if empty (spec §8.4)."""
    if not boxes:
        return None
    return max(boxes, key=bbox_area)


class ConfirmTracker:
    """Per-camera consecutive-frame confirmation before emitting detections."""

    def __init__(self, confirm_frames: int) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self.confirm_frames = confirm_frames
        self._streak: dict[str, int] = {}

    def update(self, camera_id: str, *, has_detection: bool) -> bool:
        if has_detection:
            self._streak[camera_id] = self._streak.get(camera_id, 0) + 1
        else:
            self._streak[camera_id] = 0
        return self._streak.get(camera_id, 0) >= self.confirm_frames


class CatDetector:
    """Ultralytics YOLO wrapper: COCO cat class only."""

    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tracker = ConfirmTracker(cfg.yolo_confirm_frames)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.cfg.yolo_model)

    def detect_raw(self, frame_bgr: np.ndarray) -> list[BBox]:
        """Run YOLO without confirm_frames gating (for tests or debugging)."""
        self._ensure_loaded()
        assert self._model is not None
        results = self._model.predict(
            frame_bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.yolo_conf,
            device=self.cfg.device,
            verbose=False,
        )
        boxes: list[BBox] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls[0]) != COCO_CAT_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
        return boxes

    def detect(self, frame_bgr: np.ndarray, camera_id: str) -> list[BBox]:
        """Return cat boxes only after confirm_frames consecutive detections on this camera."""
        raw = self.detect_raw(frame_bgr)
        if self._tracker.update(camera_id, has_detection=bool(raw)):
            return raw
        return []
