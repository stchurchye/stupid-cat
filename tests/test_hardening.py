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

from stupid_cat.config import ConfigError, load_config
from stupid_cat.db import Database
from stupid_cat.detector import CatDetector
from stupid_cat.ingest import MultiCameraIngest
from stupid_cat.reid import Embedder, fuse_embeddings
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
