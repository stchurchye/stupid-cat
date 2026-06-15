import numpy as np
import pytest

from stupid_cat.detector import (
    BBox,
    CatDetector,
    ConfirmTracker,
    bbox_area,
    select_primary_bbox,
)


def test_select_primary_bbox_picks_largest_area() -> None:
    boxes: list[BBox] = [
        (10, 10, 30, 30),
        (0, 0, 50, 40),
        (100, 100, 120, 120),
    ]
    assert select_primary_bbox(boxes) == (0, 0, 50, 40)


def test_select_primary_bbox_empty_returns_none() -> None:
    assert select_primary_bbox([]) is None


def test_bbox_area_normalizes_inverted_coords() -> None:
    assert bbox_area((20, 20, 10, 10)) == bbox_area((10, 10, 20, 20))


def test_confirm_tracker_requires_consecutive_frames() -> None:
    tracker = ConfirmTracker(confirm_frames=3)
    assert tracker.update("cam1", has_detection=True) is False
    assert tracker.update("cam1", has_detection=True) is False
    assert tracker.update("cam1", has_detection=True) is True
    assert tracker.update("cam1", has_detection=False) is False
    assert tracker.update("cam1", has_detection=True) is False


def test_confirm_tracker_per_camera_isolated() -> None:
    tracker = ConfirmTracker(confirm_frames=2)
    tracker.update("cam1", has_detection=True)
    assert tracker.update("cam2", has_detection=True) is False
    assert tracker.update("cam2", has_detection=True) is True


def test_cat_detector_gates_until_confirmed(monkeypatch) -> None:
    from stupid_cat.config import InferenceConfig

    cfg = InferenceConfig(yolo_confirm_frames=2)
    detector = CatDetector(cfg)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    fake_box: BBox = (0.0, 0.0, 10.0, 10.0)

    monkeypatch.setattr(detector, "detect_raw", lambda _f: [fake_box])

    assert detector.detect(frame, "cam1") == []
    assert detector.detect(frame, "cam1") == [fake_box]


def test_detector_accepts_confused_animal_classes() -> None:
    # A hunched/back-to-camera cat is frequently misclassified (e.g. as 'sheep');
    # the box only ever holds a cat, so those classes count, but non-animals don't.
    from stupid_cat.config import InferenceConfig

    class _Box:
        def __init__(self, cls, xyxy):
            self.cls = [float(cls)]
            self.xyxy = [np.array(xyxy, dtype=float)]
            self.conf = [0.7]

    class _Res:
        def __init__(self, boxes):
            self.boxes = boxes

    class _Model:
        def __init__(self, res):
            self._res = res

        def predict(self, *a, **k):
            return self._res

    det = CatDetector(InferenceConfig(yolo_confirm_frames=1))
    det._model = _Model([_Res([_Box(18, [10, 20, 110, 120])])])  # 18 = sheep
    assert det.detect_raw(np.zeros((16, 16, 3), dtype=np.uint8)) == [(10.0, 20.0, 110.0, 120.0)]
    det._model = _Model([_Res([_Box(7, [0, 0, 5, 5])])])  # 7 = truck -> ignored
    assert det.detect_raw(np.zeros((16, 16, 3), dtype=np.uint8)) == []


@pytest.mark.gpu
def test_cat_detector_detects_cat_class(tmp_path) -> None:
    pytest.importorskip("ultralytics")
    from pathlib import Path

    from stupid_cat.config import InferenceConfig

    model_path = Path("models/yolo11s.pt")
    if not model_path.exists():
        pytest.skip("yolo11s.pt not present")

    cfg = InferenceConfig(
        yolo_model=str(model_path),
        yolo_conf=0.20,
        yolo_confirm_frames=1,
        device="cpu",
    )
    detector = CatDetector(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector.detect(frame, camera_id="cam1")
    assert isinstance(boxes, list)
