"""Tests for the stability/correctness hardening pass.

Covers: FSM wall-clock watchdog (on_tick), DB crash-recovery / delete helpers,
and the threaded MultiCameraIngest (per-camera readers + idle heartbeat).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import yaml

import cv2

from stupid_cat.config import ConfigError, PreprocessConfig, load_config
from stupid_cat.db import Database
from stupid_cat.detector import CatDetector
from stupid_cat.ingest import MultiCameraIngest
from stupid_cat.reid import Embedder, build_centroid_from_refs, fuse_embeddings, ref_quality_ok
from stupid_cat.session import VisitSessionFSM

_REPO_CFG = Path(__file__).resolve().parents[1] / "config.yaml"


def _write_cfg(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(_REPO_CFG.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# --- FSM watchdog -----------------------------------------------------------


def test_on_tick_ends_active_visit_on_total_outage() -> None:
    ends: list[str] = []
    fsm = VisitSessionFSM(
        camera_ids=["cam1", "cam2"],
        enter_overlap_sec=0.05,
        exit_no_cat_sec=0.1,
        cooldown_sec=0.05,
        min_visit_sec=0.01,
    )
    fsm.on_visit_start = lambda _vid: None
    fsm.on_visit_end = lambda vid, *_a, **_k: ends.append(vid)

    t = 0.0
    for _ in range(4):
        fsm.on_frame("cam1", qualified=True, timestamp=t)
        t += 0.03
    assert fsm.state == "active"

    # A tick that is still within exit_no_cat_sec must NOT end the visit.
    fsm.on_tick(t + 0.05)
    assert fsm.state == "active"
    assert ends == []

    # A tick well past exit_no_cat_sec for every camera ends it, with no frames.
    fsm.on_tick(t + 0.5)
    assert fsm.state in ("cooldown", "idle")
    assert len(ends) == 1


def test_on_tick_noop_when_idle() -> None:
    fsm = VisitSessionFSM(
        camera_ids=["cam1"],
        enter_overlap_sec=1.0,
        exit_no_cat_sec=1.0,
        cooldown_sec=1.0,
        min_visit_sec=1.0,
    )
    fsm.on_tick(100.0)
    assert fsm.state == "idle"


# --- DB crash-recovery / delete --------------------------------------------


def test_finalize_orphan_visits_closes_open_rows(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init_schema()
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")

    ids = db.finalize_orphan_visits()
    assert ids == [vid]
    row = db.get_visit(vid)
    assert row is not None
    assert row["ended_at"] is not None
    assert row["duration_sec"] == 0
    # The finalized visit is now visible as ended.
    assert [v["id"] for v in db.list_visits(only_ended=True)] == [vid]
    # Idempotent.
    assert db.finalize_orphan_visits() == []
    db.close()


def test_delete_visit_removes_row_and_corrections(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    db.init_schema()
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:01:00+08:00",
        duration_sec=60,
        confidence=0.0,
    )
    db.correct_visit(vid, "mimi")
    assert db.list_corrections(visit_id=vid)

    db.delete_visit(vid)
    assert db.get_visit(vid) is None
    assert db.list_corrections(visit_id=vid) == []
    db.close()


# --- Threaded ingest --------------------------------------------------------


class _ListSource:
    def __init__(self, camera_id: str, count: int) -> None:
        self.camera_id = camera_id
        self._count = count

    def frames(self):
        for _ in range(self._count):
            yield np.zeros((8, 8, 3), dtype=np.uint8)


def test_threaded_ingest_reads_both_cameras() -> None:
    ingest = MultiCameraIngest(
        [_ListSource("cam1", 3), _ListSource("cam2", 3)],
        active_fps=1000.0,
        idle_fps=1000.0,
        motion_threshold=0.0,
        poll_timeout=0.02,
    )
    cameras: set[str] = set()
    n_frames = 0
    for event in ingest.events():
        if event is None:
            continue
        cameras.add(event.camera_id)
        n_frames += 1
    # Both per-camera reader threads produced frames, and the generator
    # terminated once both signalled completion (sentinels).
    assert cameras == {"cam1", "cam2"}
    assert n_frames >= 2


def test_ingest_terminates_with_many_cameras_under_backpressure() -> None:
    # Regression: 3+ readers + a tiny queue + heavy drop-oldest must still let
    # events() TERMINATE. Termination is tracked via a finished-counter, not a
    # sentinel in the lossy queue, so dropping the oldest item can never lose a
    # completion signal. (Heavy dropping may starve a camera — completeness is
    # not guaranteed under maxsize=1; termination is.)
    sources = [_ListSource(f"cam{i}", 20) for i in range(3)]
    ingest = MultiCameraIngest(
        sources,
        active_fps=100000.0,
        idle_fps=100000.0,
        motion_threshold=0.0,
        queue_maxsize=1,  # maximal backpressure / dropping
        poll_timeout=0.02,
    )
    frames = 0
    for event in ingest.events():  # reaching the end proves it terminated (no hang)
        if event is not None:
            frames += 1
    assert frames >= 1


def test_ingest_emits_heartbeat_while_a_source_is_blocked() -> None:
    release = threading.Event()

    class _BlockingSource:
        camera_id = "cam1"

        def frames(self):
            release.wait(2.0)  # simulate a stalled camera / RTSP reconnect
            yield np.zeros((8, 8, 3), dtype=np.uint8)

    ingest = MultiCameraIngest(
        [_BlockingSource()],
        active_fps=1000.0,
        idle_fps=1000.0,
        motion_threshold=0.0,
        poll_timeout=0.02,
    )
    gen = ingest.events()
    try:
        # No frame can arrive yet, so the consumer must get a heartbeat (None)
        # rather than blocking forever — this is what drives FSM timeouts.
        assert next(gen) is None
    finally:
        release.set()
        ingest.stop()


# --- FP16 (device-aware) ----------------------------------------------------


def test_embedder_fp16_only_on_cuda() -> None:
    assert Embedder(device="cpu", fp16=True)._use_half is False
    assert Embedder(device="cuda:0", fp16=True)._use_half is True
    assert Embedder(device="cuda:0", fp16=False)._use_half is False


def test_detector_fp16_only_on_cuda() -> None:
    from stupid_cat.config import InferenceConfig

    assert CatDetector(InferenceConfig(device="cpu", fp16=True))._use_half is False
    assert CatDetector(InferenceConfig(device="cuda:0", fp16=True))._use_half is True


# --- weighted_median actually weights ---------------------------------------


def test_weighted_median_respects_weights() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    toward_a = fuse_embeddings([a, b], [3.0, 1.0], mode="weighted_median")
    toward_b = fuse_embeddings([a, b], [1.0, 3.0], mode="weighted_median")
    # The heavier-weighted embedding dominates the per-dimension median, so the
    # fused direction leans toward it (the old int-rounding code ignored 0.5/1.5).
    assert toward_a[0] > toward_a[1]
    assert toward_b[1] > toward_b[0]


# --- camera enabled flag ----------------------------------------------------


def test_disabled_camera_excluded(tmp_path: Path) -> None:
    cfg_path = _write_cfg(tmp_path, lambda d: d["cameras"][1].__setitem__("enabled", False))
    cfg = load_config(cfg_path)
    assert [c.id for c in cfg.cameras if c.enabled] == ["cam1"]


def test_all_cameras_disabled_is_error(tmp_path: Path) -> None:
    def _disable_all(d: dict) -> None:
        for cam in d["cameras"]:
            cam["enabled"] = False

    cfg_path = _write_cfg(tmp_path, _disable_all)
    with pytest.raises(ConfigError, match="at least one camera must be enabled"):
        load_config(cfg_path)


# --- reference-image quality gate (keeps dark IR, drops blank) ---------------


def test_ref_quality_gate_keeps_dark_textured_drops_blank() -> None:
    blank = np.full((40, 40, 3), 3, dtype=np.uint8)  # uniform near-black
    bright_blank = np.full((40, 40, 3), 252, dtype=np.uint8)  # uniform near-white
    dark_textured = np.full((40, 40, 3), 3, dtype=np.uint8)
    dark_textured[8:32, 8:32] = 40  # a dark cat with real texture
    tiny = np.full((8, 8, 3), 100, dtype=np.uint8)

    assert ref_quality_ok(blank) is False
    assert ref_quality_ok(bright_blank) is False
    assert ref_quality_ok(dark_textured) is True  # low mean but real dynamic range
    assert ref_quality_ok(tiny) is False  # too small


class _FakeEmbedder:
    def embed(self, frame: np.ndarray) -> np.ndarray:
        vec = np.zeros(8, dtype=np.float32)
        vec[0] = 1.0
        return vec


def test_build_centroid_keeps_dark_refs(tmp_path: Path) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(5):
        img = np.full((40, 40, 3), 3, dtype=np.uint8)
        img[6:34, 6:34] = 25 + i  # dark but textured
        cv2.imwrite(str(refs / f"r{i}.jpg"), img)
    centroid = build_centroid_from_refs(_FakeEmbedder(), refs, PreprocessConfig(), min_refs=5)
    assert centroid is not None  # dark IR refs must NOT be rejected as blank


def test_build_centroid_rejects_all_blank(tmp_path: Path) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(5):
        cv2.imwrite(str(refs / f"b{i}.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
    centroid = build_centroid_from_refs(_FakeEmbedder(), refs, PreprocessConfig(), min_refs=5)
    assert centroid is None  # all uniform/blank -> skipped -> below min_refs
