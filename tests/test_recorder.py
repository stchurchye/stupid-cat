from pathlib import Path

import numpy as np
import pytest

from stupid_cat.recorder import VisitRecorder, visit_recording_filename


def test_recorder_stops_writing_after_max_seconds(tmp_path: Path) -> None:
    recorder = VisitRecorder(
        tmp_path / "recordings",
        max_seconds=30,
        fps=10,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    recorder.start("visit-1", frame)
    for _ in range(350):
        recorder.write_frame(frame)
    path = recorder.stop()
    assert path is not None
    assert path.name == "visit-1.mp4"
    assert recorder.frames_written == 300


def test_recorder_stop_without_start_returns_none(tmp_path: Path) -> None:
    recorder = VisitRecorder(tmp_path, max_seconds=30)
    assert recorder.stop() is None


def test_visit_recording_filename_primary_and_secondary() -> None:
    assert visit_recording_filename("v1", "cam1", primary_camera="cam1") == "v1.mp4"
    assert visit_recording_filename("v1", "cam2", primary_camera="cam1") == "v1_cam2.mp4"


def test_recorder_custom_filename(tmp_path: Path) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    recorder = VisitRecorder(tmp_path / "rec", max_seconds=30, fps=10)
    path = recorder.start("visit-1", frame, filename="visit-1_cam2.mp4")
    assert path.name == "visit-1_cam2.mp4"
    recorder.stop(reencode=False)


def test_recorder_wall_clock_cap_stops_writing(tmp_path: Path, monkeypatch) -> None:
    import stupid_cat.recorder as rec_mod

    clock = [1000.0]
    monkeypatch.setattr(rec_mod.time, "monotonic", lambda: clock[0])
    # Huge frame budget so only the wall-clock cap can stop writing.
    recorder = VisitRecorder(tmp_path / "rec", max_seconds=30, fps=10000)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    recorder.start("v", frame)  # captures _start_mono = 1000.0

    assert recorder.write_frame(frame) is True  # still within 30s
    clock[0] = 1031.0  # 31s elapsed -> past max_seconds
    assert recorder.write_frame(frame) is False
    recorder.stop(reencode=False)


def test_replace_with_retry_succeeds(tmp_path) -> None:
    from stupid_cat.recorder import _replace_with_retry

    src = tmp_path / "a.tmp"
    src.write_bytes(b"x")
    dst = tmp_path / "b.mp4"
    assert _replace_with_retry(src, dst, attempts=2, delay=0) is True
    assert dst.read_bytes() == b"x"
    assert not src.exists()


def test_replace_with_retry_gives_up_and_cleans_temp(tmp_path) -> None:
    from stupid_cat.recorder import _replace_with_retry

    src = tmp_path / "a.tmp"
    src.write_bytes(b"x")
    dst = tmp_path / "dstdir"  # non-empty dir -> replace raises OSError every time
    dst.mkdir()
    (dst / "keep").write_bytes(b"")
    assert _replace_with_retry(src, dst, attempts=2, delay=0) is False
    assert not src.exists()  # temp cleaned up after giving up
