import numpy as np

from stupid_cat.ingest import VideoFileSource, motion_score, rate_limited


def test_motion_score_increases_on_change() -> None:
    prev = np.zeros((10, 10), dtype=np.uint8)
    curr = prev.copy()
    curr[5, 5] = 255
    assert motion_score(prev, curr) > motion_score(prev, prev)


def test_rate_limited_respects_fps(tmp_path, monkeypatch) -> None:
    import stupid_cat.ingest as ingest_mod

    class FakeSource:
        camera_id = "cam1"

        def frames(self):
            for _ in range(10):
                yield np.zeros((8, 8, 3), dtype=np.uint8)

    times = iter([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0])
    monkeypatch.setattr(ingest_mod.time, "monotonic", lambda: next(times))

    events = list(
        rate_limited(FakeSource(), active_fps=2.0, idle_fps=2.0, motion_threshold=0)
    )
    assert len(events) >= 3
    assert all(e.camera_id == "cam1" for e in events)


def test_rtsp_backoff_grows_and_caps() -> None:
    from stupid_cat.ingest import RtspSource

    s = RtspSource("cam1", "rtsp://x", reconnect_delay_sec=5.0, max_reconnect_delay_sec=60.0)
    assert s._backoff_delay(0) == 5.0
    assert s._backoff_delay(1) == 10.0
    assert s._backoff_delay(2) == 20.0
    assert s._backoff_delay(10) == 60.0  # capped at max
