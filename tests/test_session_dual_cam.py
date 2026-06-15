
from stupid_cat.session import VisitSessionFSM


def test_dual_cam_keeps_visit_while_either_camera_sees_cat() -> None:
    """Visit should continue while any camera still qualifies within exit window."""
    ended: list[bool] = []
    fsm = VisitSessionFSM(
        camera_ids=["cam1", "cam2"],
        enter_overlap_sec=0.05,
        exit_no_cat_sec=0.2,
        cooldown_sec=0.1,
        min_visit_sec=0.01,
    )
    fsm.on_visit_start = lambda _vid: None
    fsm.on_visit_end = lambda *_args, **_kwargs: ended.append(True)

    t = 0.0
    for _ in range(4):
        fsm.on_frame("cam1", qualified=True, timestamp=t)
        t += 0.02
    assert fsm.state == "active"

    # cam1 lost, cam2 still sees cat — visit must stay active
    for _ in range(5):
        fsm.on_frame("cam1", qualified=False, timestamp=t)
        fsm.on_frame("cam2", qualified=True, timestamp=t)
        t += 0.03
    assert fsm.state == "active"
    assert ended == []

    # both cameras lost for longer than exit_no_cat_sec
    for _ in range(8):
        fsm.on_frame("cam1", qualified=False, timestamp=t)
        fsm.on_frame("cam2", qualified=False, timestamp=t)
        t += 0.03
    assert ended == [True]
