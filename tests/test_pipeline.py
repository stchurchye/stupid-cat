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
    # This fixture runs a single camera; keep record_cameras consistent with it
    # (the shipped config records both cam1+cam2).
    data["recorder"]["record_cameras"] = ["cam1"]
    # Test frames are 640x480; match the camera's calibration resolution so the
    # ROI (authored in stream coords) maps 1:1 onto the frame (spec §7.3).
    data["cameras"][0]["stream_width"] = 640
    data["cameras"][0]["stream_height"] = 480
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


def test_visit_row_exists_during_active_visit(fast_pipeline: Pipeline) -> None:
    # Persisting at visit start means an active (un-ended) row exists, so a crash
    # mid-visit is recoverable instead of losing the visit entirely.
    t = 0.0
    for _ in range(5):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    active_id = fast_pipeline.fsm.visit_id
    assert active_id is not None
    row = fast_pipeline.db.get_visit(active_id)
    assert row is not None
    assert row["ended_at"] is None
    # Not yet finalized, so it must not appear in the ended-only timeline.
    assert fast_pipeline.db.list_visits(only_ended=True) == []


def test_recover_orphan_visits_on_construction(fast_pipeline: Pipeline, tmp_path: Path) -> None:
    # Simulate a crash: an open visit row with no ended_at.
    orphan = fast_pipeline.db.create_visit(
        cat_id="unknown", started_at="2026-06-02T10:00:00+08:00"
    )
    assert fast_pipeline.db.get_visit(orphan)["ended_at"] is None

    # Re-running recovery (as __init__ does) finalizes it.
    fast_pipeline._recover_orphan_visits()
    row = fast_pipeline.db.get_visit(orphan)
    assert row["ended_at"] is not None


def test_process_frame_reports_work_only_when_embedding(fast_pipeline: Pipeline) -> None:
    # A no-cat frame does no Re-ID work -> falsy (so it won't reset the error
    # streak); frames during an active visit embed -> truthy.
    fast_pipeline.detector.in_roi = False
    assert not fast_pipeline._process_frame(FrameEvent("cam1", _frame(), 0.0))

    fast_pipeline.detector.in_roi = True
    did_work = False
    t = 0.0
    for _ in range(6):
        t += 0.05
        did_work = bool(fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))) or did_work
    assert did_work is True


def test_disk_guard_skips_recording_when_low(fast_pipeline: Pipeline, monkeypatch) -> None:
    import collections

    import stupid_cat.pipeline as pipeline_mod

    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(pipeline_mod.shutil, "disk_usage", lambda _p: usage(100, 100, 0))
    assert fast_pipeline._disk_ok() is False
    # min_free_mb <= 0 disables the guard entirely.
    fast_pipeline.cfg.recorder.min_free_mb = 0
    assert fast_pipeline._disk_ok() is True


def test_roi_normalized_is_resolution_independent(fast_pipeline: Pipeline) -> None:
    # 0..1 coords scale by the frame size and map to the same RELATIVE box at any
    # resolution (the green box never shifts across main/substream).
    fast_pipeline.roi_by_camera["cam1"] = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.4], [0.1, 0.4]]
    roi_a = fast_pipeline._roi_for_frame("cam1", 640, 480)
    assert roi_a[0] == [64.0, 48.0] and roi_a[2] == [320.0, 192.0]
    roi_b = fast_pipeline._roi_for_frame("cam1", 1280, 720)
    assert roi_b[0] == [128.0, 72.0] and roi_b[2] == [640.0, 288.0]


def test_roi_scaled_to_actual_frame_resolution(fast_pipeline: Pipeline) -> None:
    # Fixture calibrates the camera at 640x480.
    roi_full = fast_pipeline._roi_for_frame("cam1", 640, 480)
    roi_half = fast_pipeline._roi_for_frame("cam1", 320, 240)
    # At the calibration resolution the ROI is unchanged; at half resolution
    # every coordinate is halved (spec §7.3 substream support).
    assert roi_half[0][0] == roi_full[0][0] / 2
    assert roi_half[2][1] == roi_full[2][1] / 2


def test_set_centroid_is_copy_on_write(fast_pipeline: Pipeline) -> None:
    fast_pipeline.centroids = {"a": np.ones(4, dtype=np.float32)}
    snapshot = fast_pipeline.centroids
    fast_pipeline._set_centroid("b", np.zeros(4, dtype=np.float32))
    # The live dict gained the new key; the captured snapshot is untouched, so a
    # concurrent reader iterating the snapshot can never see a mid-write mutation.
    assert "b" in fast_pipeline.centroids
    assert "b" not in snapshot
    assert fast_pipeline.centroids is not snapshot


def test_correct_visit_appends_ref(fast_pipeline: Pipeline, tmp_path: Path) -> None:
    visit_id = "visit-test-1"
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    fast_pipeline._save_correction_crops(visit_id, {"cam1": crop})
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

    assert fast_pipeline._load_correction_crops(visit_id)
    fast_pipeline.correct_visit(visit_id, "mimi")
    ref_path = tmp_path / "data" / "cats" / "mimi" / "refs" / f"{visit_id}_cam1.jpg"
    assert ref_path.exists()
    assert fast_pipeline._load_correction_crops(visit_id) == {}
    row = fast_pipeline.db.get_visit(visit_id)
    assert row is not None
    assert row["cat_id"] == "mimi"


def test_correct_visit_appends_dual_camera_refs(fast_pipeline: Pipeline, tmp_path: Path) -> None:
    visit_id = "visit-dual-1"
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    fast_pipeline._save_correction_crops(
        visit_id,
        {"cam1": crop, "cam2": crop + 1},
    )
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

    fast_pipeline.correct_visit(visit_id, "mimi")
    refs_dir = tmp_path / "data" / "cats" / "mimi" / "refs"
    assert (refs_dir / f"{visit_id}_cam1.jpg").exists()
    assert (refs_dir / f"{visit_id}_cam2.jpg").exists()


def test_last_qualified_iso_seeded_at_visit_start(fast_pipeline: Pipeline) -> None:
    # Seeded at start so a visit that loses the cat right after entry still
    # reports presence-based duration, not the exit-timeout moment.
    t = 0.0
    for _ in range(4):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    assert fast_pipeline.fsm.state == "active"
    assert fast_pipeline._last_qualified_iso is not None


def test_multi_cat_debounced_against_single_frame_split(fast_pipeline: Pipeline) -> None:
    class _SplitOnceDetector:
        in_roi = True
        calls = 0

        def detect(self, frame, camera_id):
            if not self.in_roi:
                return []
            self.calls += 1
            if self.calls <= 2:  # a transient 2nd box (YOLO split), not sustained
                return [(150.0, 150.0, 250.0, 250.0), (300.0, 150.0, 400.0, 250.0)]
            return [(150.0, 150.0, 250.0, 250.0)]

    fast_pipeline.detector = _SplitOnceDetector()
    t = 0.0
    for _ in range(12):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    fast_pipeline.detector.in_roi = False
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    visit = fast_pipeline.db.list_visits(only_ended=True)[0]
    assert visit["max_cats"] == 1  # not flagged from a one/two-frame split
    assert visit["multi_cat"] is False


def test_digging_samples_only_while_cat_present(fast_pipeline: Pipeline) -> None:
    t = 0.0
    for _ in range(8):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    n = len(fast_pipeline._motion_samples)
    assert n >= 1
    fast_pipeline.detector.in_roi = False
    for _ in range(5):  # no-cat tail (visit still active briefly)
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    # The empty post-departure tail must NOT add digging samples (else it would
    # dilute the late-phase mean toward 'pee').
    assert len(fast_pipeline._motion_samples) == n


def test_correct_visit_keeps_old_ref_on_write_failure(fast_pipeline: Pipeline, tmp_path: Path, monkeypatch) -> None:
    import stupid_cat.pipeline as pmod

    vid = "v-writefail"
    fast_pipeline._save_correction_crops(vid, {"cam1": np.zeros((100, 100, 3), dtype=np.uint8)})
    fast_pipeline.db.create_visit(visit_id=vid, cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    fast_pipeline.db.end_visit(vid, cat_id="unknown", ended_at="2026-06-02T10:01:00+08:00",
                               duration_sec=60, confidence=0.0)
    fast_pipeline.correct_visit(vid, "mimi")
    mimi_ref = tmp_path / "data" / "cats" / "mimi" / "refs" / f"{vid}_cam1.jpg"
    assert mimi_ref.exists()

    # Re-correct to cat2 but the ref write fails: the old ref must survive.
    monkeypatch.setattr(pmod.cv2, "imwrite", lambda *a, **k: False)
    fast_pipeline.correct_visit(vid, "cat2")
    assert mimi_ref.exists()
    assert not (tmp_path / "data" / "cats" / "cat2" / "refs" / f"{vid}_cam1.jpg").exists()


def test_correct_visit_unknown_roundtrip_preserves_crop(fast_pipeline: Pipeline, tmp_path: Path) -> None:
    vid = "v-roundtrip"
    fast_pipeline._save_correction_crops(vid, {"cam1": np.zeros((100, 100, 3), dtype=np.uint8)})
    fast_pipeline.db.create_visit(visit_id=vid, cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    fast_pipeline.db.end_visit(vid, cat_id="unknown", ended_at="2026-06-02T10:01:00+08:00",
                               duration_sec=60, confidence=0.0)
    fast_pipeline.correct_visit(vid, "mimi")
    fast_pipeline.correct_visit(vid, "unknown")  # must re-persist the crop
    fast_pipeline.correct_visit(vid, "cat2")     # must still be able to learn
    assert (tmp_path / "data" / "cats" / "cat2" / "refs" / f"{vid}_cam1.jpg").exists()
    assert not (tmp_path / "data" / "cats" / "mimi" / "refs" / f"{vid}_cam1.jpg").exists()


def test_classify_waste_heuristic(fast_pipeline: Pipeline) -> None:
    w = fast_pipeline.cfg.waste
    w.enabled = True
    # poop: long visit + heavy late digging
    assert fast_pipeline._classify_waste(w.poop_min_duration_sec + 10, w.dig_motion_threshold + 5)[0] == "poop"
    # pee: short visit + little digging
    assert fast_pipeline._classify_waste(w.pee_max_duration_sec - 10, w.dig_motion_threshold - 5)[0] == "pee"
    # ambiguous combinations -> unknown
    assert fast_pipeline._classify_waste(w.poop_min_duration_sec + 10, 0)[0] == "unknown"
    assert fast_pipeline._classify_waste(w.pee_max_duration_sec - 10, w.dig_motion_threshold + 50)[0] == "unknown"


def test_multi_cat_visit_is_flagged(fast_pipeline: Pipeline) -> None:
    class _TwoCatDetector:
        in_roi = True

        def detect(self, frame, camera_id):
            if not self.in_roi:
                return []
            return [(150.0, 150.0, 250.0, 250.0), (300.0, 150.0, 400.0, 250.0)]

    fast_pipeline.detector = _TwoCatDetector()
    t = 0.0
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    fast_pipeline.detector.in_roi = False
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    visit = fast_pipeline.db.list_visits(only_ended=True)[0]
    assert visit["max_cats"] == 2
    assert visit["multi_cat"] is True


def test_multi_cat_forces_unknown_identity(fast_pipeline: Pipeline) -> None:
    # A matching centroid is present, so a single cat WOULD be identified; a
    # multi-cat visit must still be forced to unknown (don't guess a blend).
    centroid = np.zeros(1280, dtype=np.float32)
    centroid[0] = 1.0  # matches FakeEmbedder's output
    fast_pipeline.centroids = {"mimi": centroid}

    class _TwoCatDetector:
        in_roi = True

        def detect(self, frame, camera_id):
            if not self.in_roi:
                return []
            return [(150.0, 150.0, 250.0, 250.0), (300.0, 150.0, 400.0, 250.0)]

    fast_pipeline.detector = _TwoCatDetector()
    t = 0.0
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    fast_pipeline.detector.in_roi = False
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    visit = fast_pipeline.db.list_visits(only_ended=True)[0]
    assert visit["multi_cat"] is True
    assert visit["cat_id"] == "unknown"  # forced despite the matching centroid
    assert visit["confidence"] == 0.0


def test_visit_active_flag_tracks_visit(fast_pipeline: Pipeline) -> None:
    assert not fast_pipeline._visit_active.is_set()
    t = 0.0
    for _ in range(6):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    assert fast_pipeline._visit_active.is_set()  # full-fps recording while active

    fast_pipeline.detector.in_roi = False
    for _ in range(10):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    assert not fast_pipeline._visit_active.is_set()  # cleared once the visit ends


def test_duration_uses_last_qualified_not_exit_timeout(fast_pipeline: Pipeline, monkeypatch) -> None:
    import stupid_cat.pipeline as pmod
    from datetime import datetime, timedelta, timezone

    clock = {"t": 0}

    def _fake_iso() -> str:
        return (datetime(2026, 6, 15, tzinfo=timezone.utc) + timedelta(seconds=clock["t"])).isoformat(timespec="seconds")

    monkeypatch.setattr(pmod, "_iso_now", _fake_iso)

    ev = 0.0
    for i in range(6):  # cat present, wall-clock seconds 0..5
        clock["t"] = i
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), ev))
        ev += 0.05
    fast_pipeline.detector.in_roi = False
    for j in range(10):  # cat gone; wall clock jumps far ahead during the no-cat tail
        clock["t"] = 100 + j
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), ev))
        ev += 0.05

    visit = fast_pipeline.db.list_visits(only_ended=True)[0]
    # Duration tracks actual presence (~5s), NOT the timeout moment (~100s).
    assert visit["duration_sec"] <= 10


def test_recorrection_moves_ref_between_cats(fast_pipeline: Pipeline, tmp_path: Path) -> None:
    visit_id = "visit-recorr"
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    fast_pipeline._save_correction_crops(visit_id, {"cam1": crop})
    fast_pipeline.db.create_visit(visit_id=visit_id, cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    fast_pipeline.db.end_visit(visit_id, cat_id="unknown", ended_at="2026-06-02T10:01:00+08:00",
                               duration_sec=60, confidence=0.0)

    fast_pipeline.correct_visit(visit_id, "mimi")
    mimi_ref = tmp_path / "data" / "cats" / "mimi" / "refs" / f"{visit_id}_cam1.jpg"
    assert mimi_ref.exists()

    # Re-correct to a different cat: the ref must MOVE, not be orphaned under mimi.
    fast_pipeline.correct_visit(visit_id, "cat2")
    cat2_ref = tmp_path / "data" / "cats" / "cat2" / "refs" / f"{visit_id}_cam1.jpg"
    assert cat2_ref.exists()
    assert not mimi_ref.exists()
    assert fast_pipeline.db.get_visit(visit_id)["cat_id"] == "cat2"


def test_preview_stale_frame_returns_none(fast_pipeline: Pipeline, monkeypatch) -> None:
    import stupid_cat.pipeline as pmod

    with fast_pipeline._preview_lock:
        fast_pipeline._preview_frames["cam1"] = np.zeros((48, 64, 3), dtype=np.uint8)
        fast_pipeline._preview_ts["cam1"] = 1000.0

    monkeypatch.setattr(pmod, "_mono_now", lambda: 1005.0)  # 5s old -> fresh
    assert fast_pipeline.get_preview_jpeg("cam1") is not None
    monkeypatch.setattr(pmod, "_mono_now", lambda: 1020.0)  # 20s old -> stale
    assert fast_pipeline.get_preview_jpeg("cam1") is None


def test_orphan_recovery_relinks_secondary_clip(fast_pipeline: Pipeline) -> None:
    rec_dir = fast_pipeline.recordings_dir
    rec_dir.mkdir(parents=True, exist_ok=True)
    vid = fast_pipeline.db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    # Only a secondary-camera clip exists (primary never recorded before the crash).
    (rec_dir / f"{vid}_cam2.mp4").write_bytes(b"\x00\x00")

    fast_pipeline._recover_orphan_visits()
    row = fast_pipeline.db.get_visit(vid)
    assert row["ended_at"] is not None
    assert row["recording_path"] is not None
    assert row["recording_path"].endswith(f"{vid}_cam2.mp4")


def test_dual_camera_recording_writes_both_clips(tmp_path: Path) -> None:
    repo_cfg = Path(__file__).resolve().parents[1] / "config.yaml"
    data = yaml.safe_load(repo_cfg.read_text(encoding="utf-8"))
    data["session"]["enter_overlap_sec"] = 0.05
    data["session"]["exit_no_cat_sec"] = 0.1
    data["session"]["cooldown_sec"] = 0.05
    data["session"]["min_visit_sec"] = 0.01
    data["recorder"]["record_cameras"] = ["cam1", "cam2"]
    data["recorder"]["min_free_mb"] = 0
    for cam in data["cameras"]:
        cam["stream_width"] = 640
        cam["stream_height"] = 480
    cfg_path = tmp_path / "config.yaml"
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
    pipeline.cfg.recorder.enabled = True

    t = 0.0
    for _ in range(10):
        pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        pipeline._process_frame(FrameEvent("cam2", _frame(), t))
        t += 0.05
    pipeline.detector.in_roi = False
    for _ in range(20):
        pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        pipeline._process_frame(FrameEvent("cam2", _frame(), t))
        t += 0.05

    visit = pipeline.db.list_visits(only_ended=True)[0]
    rec_dir = pipeline.recordings_dir
    assert (rec_dir / f"{visit['id']}.mp4").exists()
    assert (rec_dir / f"{visit['id']}_cam2.mp4").exists()


def test_dual_camera_visit_saves_single_correction_crop(tmp_path: Path) -> None:
    # A dual-camera visit must save ONE correction crop (one appearance = one
    # ref), not one per camera, or the unweighted centroid double-counts it.
    repo_cfg = Path(__file__).resolve().parents[1] / "config.yaml"
    data = yaml.safe_load(repo_cfg.read_text(encoding="utf-8"))
    data["session"]["enter_overlap_sec"] = 0.05
    data["session"]["exit_no_cat_sec"] = 0.1
    data["session"]["cooldown_sec"] = 0.05
    data["session"]["min_visit_sec"] = 0.01
    data["recorder"]["enabled"] = False
    for cam in data["cameras"]:
        cam["stream_width"] = 640
        cam["stream_height"] = 480
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(cfg_path)
    pipeline = Pipeline(
        cfg, db=Database(tmp_path / "t.db"), data_dir=tmp_path / "data",
        detector=FakeDetector(), embedder=FakeEmbedder(),
    )

    # Textured crop so it passes ref_quality_ok (a flat crop would be rejected).
    textured = np.zeros((480, 640, 3), dtype=np.uint8)
    textured[150:250, 150:250] = (np.arange(100).reshape(100, 1) % 200).astype(np.uint8)[:, :, None]

    t = 0.0
    for _ in range(10):
        pipeline._process_frame(FrameEvent("cam1", textured, t))
        pipeline._process_frame(FrameEvent("cam2", textured, t))
        t += 0.05
    pipeline.detector.in_roi = False
    for _ in range(20):
        pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        pipeline._process_frame(FrameEvent("cam2", _frame(), t))
        t += 0.05

    crop_dir = tmp_path / "data" / "correction_crops"
    saved = list(crop_dir.glob("*.jpg")) if crop_dir.is_dir() else []
    assert len(saved) == 1


def test_recording_continues_without_detection(fast_pipeline: Pipeline) -> None:
    """Clips should keep writing while the visit is active even if YOLO misses the cat."""
    fast_pipeline.cfg.recorder.enabled = True
    fast_pipeline.cfg.recorder.min_free_mb = 0
    fast_pipeline.fsm.exit_no_cat_sec = 999.0

    t = 0.0
    for _ in range(6):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05
    assert fast_pipeline.fsm.state == "active"

    fast_pipeline.detector.in_roi = False
    for _ in range(20):
        fast_pipeline._process_frame(FrameEvent("cam1", _frame(), t))
        t += 0.05

    rec = fast_pipeline._visit_recorders.get("cam1")
    assert rec is not None
    assert rec.frames_written >= 21
