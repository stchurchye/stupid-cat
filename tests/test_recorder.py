from pathlib import Path

import numpy as np
import pytest

from stupid_cat.recorder import VisitRecorder


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
