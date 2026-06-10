from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from stupid_cat.config import load_config
from stupid_cat.db import Database
from stupid_cat.ingest import FrameEvent
from stupid_cat.pipeline import Pipeline, _duration_seconds_wall


class FakeDetector:
    def __init__(self) -> None:
        self.in_roi = True

    def detect(self, frame: np.ndarray, camera_id: str) -> list[tuple[float, float, float, float]]:
        if not self.in_roi:
            return []
        return [(150.0, 150.0, 250.0, 250.0)]


class FakeEmbedder:
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        vec = np.zeros(1280, dtype=np.float32)
        vec[0] = 1.0
        return vec


@pytest.fixture
def fast_pipeline(tmp_path: Path) -> Pipeline:
    repo_cfg = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(repo_cfg)
    cfg_path = tmp_path / "config.yaml"
    data = yaml.safe_load(repo_cfg.read_text(encoding="utf-8"))
    data["session"]["enter_overlap_sec"] = 0.05
    data["session"]["exit_no_cat_sec"] = 0.1
    data["session"]["cooldown_sec"] = 0.05
    data["session"]["min_visit_sec"] = 0.01
    data["cameras"] = [data["cameras"][0]]
    cfg_path.write_text(yaml.dump(data), encoding="utf-8")

    cfg = load_config(cfg_path)
    db = Database(tmp_path / "test.db")
    pipeline = Pipeline(
        cfg,
        db=db,
        data_dir=tmp_path / "data",
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
    )
    return pipeline


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_visit_end_writes_db(fast_pipeline: Pipeline) -> None:
    t = 0.0
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    fast_pipeline.detector.in_roi = False
    for _ in range(20):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    visits = fast_pipeline.db.list_visits(only_ended=True)
    assert len(visits) == 1
    assert visits[0]["duration_sec"] is not None


def test_discard_clears_recording_state(fast_pipeline: Pipeline) -> None:
    fast_pipeline.cfg.recorder.enabled = True
    fast_pipeline.fsm.min_visit_sec = 10.0

    t = 0.0
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    fast_pipeline.detector.in_roi = False
    for _ in range(20):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    assert fast_pipeline._recording_path is None
    assert fast_pipeline.fsm.state in ("idle", "cooldown")
    assert fast_pipeline.db.list_visits(only_ended=True) == []


def test_visit_duration_uses_wall_clock(fast_pipeline: Pipeline) -> None:
    t = 0.0
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    fast_pipeline.detector.in_roi = False
    for _ in range(20):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    visit = fast_pipeline.db.list_visits(only_ended=True)[0]
    expected = _duration_seconds_wall(visit["started_at"], visit["ended_at"])
    assert visit["duration_sec"] == expected


def test_correct_visit_appends_ref(fast_pipeline: Pipeline, tmp_path: Path) -> None:
    visit_id = "visit-test-1"
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    fast_pipeline._save_correction_crop(visit_id, crop)
    fast_pipeline.db.create_visit(
        visit_id=visit_id,
        cat_id="unknown",
        started_at="2026-06-02T10:00:00+08:00",
    )
    fast_pipeline.db.end_visit(
        visit_id,
        cat_id="unknown",
        ended_at="2026-06-02T10:01:00+08:00",
        duration_sec=60,
        confidence=0.0,
    )

    assert fast_pipeline._load_correction_crop(visit_id) is not None
    fast_pipeline.correct_visit(visit_id, "mimi")
    ref_path = tmp_path / "data" / "cats" / "mimi" / "refs" / f"{visit_id}.jpg"
    assert ref_path.exists()
    assert fast_pipeline._load_correction_crop(visit_id) is None
    row = fast_pipeline.db.get_visit(visit_id)
    assert row is not None
    assert row["cat_id"] == "mimi"
