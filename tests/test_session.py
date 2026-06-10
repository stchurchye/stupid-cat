from stupid_cat.session import VisitSessionFSM


def _run_lifecycle() -> tuple[list[str], list[str]]:
    starts: list[str] = []
    ends: list[str] = []

    fsm = VisitSessionFSM(
        camera_ids=["cam1", "cam2"],
        enter_overlap_sec=0.1,
        exit_no_cat_sec=0.2,
        cooldown_sec=0.1,
        min_visit_sec=0.05,
    )
    fsm.on_visit_start = starts.append
    fsm.on_visit_end = lambda vid, _started, _ended, _dur, discard=False: ends.append(
        f"{vid}:{'discard' if discard else 'ok'}"
    )

    t = 0.0
    for _ in range(5):
        fsm.on_frame("cam1", qualified=True, timestamp=t)
        t += 0.03

    for _ in range(8):
        fsm.on_frame("cam1", qualified=False, timestamp=t)
        fsm.on_frame("cam2", qualified=False, timestamp=t)
        t += 0.03

    fsm.on_frame("cam1", qualified=True, timestamp=t)
    t += 0.15
    fsm.on_frame("cam1", qualified=True, timestamp=t)
    for _ in range(5):
        fsm.on_frame("cam1", qualified=True, timestamp=t)
        t += 0.03

    return starts, ends


def test_visit_lifecycle() -> None:
    starts, ends = _run_lifecycle()
    assert len(starts) == 2
    assert len(ends) == 1
    assert ends[0].endswith(":ok")


def test_pause_ends_visit_without_cooldown() -> None:
    starts: list[str] = []
    ends: list[str] = []
    fsm = VisitSessionFSM(
        camera_ids=["cam1"],
        enter_overlap_sec=0.05,
        exit_no_cat_sec=10.0,
        cooldown_sec=0.5,
        min_visit_sec=0.01,
    )
    fsm.on_visit_start = starts.append
    fsm.on_visit_end = lambda vid, *_a, **_k: ends.append(vid)

    t = 0.0
    for _ in range(4):
        fsm.on_frame("cam1", qualified=True, timestamp=t)
        t += 0.02
    assert len(starts) == 1

    fsm.pause(t)
    assert len(ends) == 1
    assert fsm.state == "idle"

    fsm.on_frame("cam1", qualified=True, timestamp=t)
    t += 0.02
    fsm.on_frame("cam1", qualified=True, timestamp=t)
    assert len(starts) == 1


def test_short_visit_discarded() -> None:
    discarded: list[bool] = []
    fsm = VisitSessionFSM(
        camera_ids=["cam1"],
        enter_overlap_sec=0.05,
        exit_no_cat_sec=0.05,
        cooldown_sec=0.05,
        min_visit_sec=1.0,
    )
    fsm.on_visit_end = lambda _vid, _s, _e, _d, discard: discarded.append(discard)

    t = 0.0
    for _ in range(4):
        fsm.on_frame("cam1", qualified=True, timestamp=t)
        t += 0.02
    fsm.on_frame("cam1", qualified=False, timestamp=t)
    t += 0.08
    for _ in range(4):
        fsm.on_frame("cam1", qualified=False, timestamp=t)
        t += 0.02

    assert discarded == [True]
