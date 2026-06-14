"""Global visit FSM for dual-camera ROI events (spec §8)."""

from __future__ import annotations

import uuid
from collections.abc import Callable

VisitStartCallback = Callable[[str], None]
VisitEndCallback = Callable[[str, float, float, float, bool], None]


class VisitSessionFSM:
    def __init__(
        self,
        *,
        camera_ids: list[str],
        enter_overlap_sec: float,
        exit_no_cat_sec: float,
        cooldown_sec: float,
        min_visit_sec: float,
    ) -> None:
        if not camera_ids:
            raise ValueError("camera_ids must not be empty")
        self.camera_ids = list(camera_ids)
        self.enter_overlap_sec = enter_overlap_sec
        self.exit_no_cat_sec = exit_no_cat_sec
        self.cooldown_sec = cooldown_sec
        self.min_visit_sec = min_visit_sec

        self.state = "idle"
        self.enter_accumulator = 0.0
        self.cooldown_until: float | None = None
        self.visit_id: str | None = None
        self.started_at: float | None = None
        self._last_ts: float | None = None
        self._last_qualified_at: dict[str, float] = {cid: 0.0 for cid in self.camera_ids}
        self._last_frame_qualified: dict[str, bool] = {cid: False for cid in self.camera_ids}

        self.on_visit_start: VisitStartCallback | None = None
        self.on_visit_end: VisitEndCallback | None = None

    def on_frame(self, camera_id: str, *, qualified: bool, timestamp: float) -> None:
        if camera_id not in self._last_qualified_at:
            raise ValueError(f"unknown camera_id: {camera_id}")

        dt = 0.0 if self._last_ts is None else max(0.0, timestamp - self._last_ts)
        self._last_ts = timestamp

        if self.state == "cooldown":
            if self.cooldown_until is not None and timestamp < self.cooldown_until:
                return
            self.state = "idle"
            self.enter_accumulator = 0.0
            self.cooldown_until = None

        self._last_frame_qualified[camera_id] = qualified
        if qualified:
            self._last_qualified_at[camera_id] = timestamp

        if self.state == "active":
            self._maybe_end_visit(timestamp)
            return

        if self.state != "idle":
            return

        any_qualified = any(self._last_frame_qualified.values())
        if any_qualified:
            self.enter_accumulator += dt
        else:
            self.enter_accumulator = 0.0

        if self.enter_accumulator >= self.enter_overlap_sec:
            self._start_visit(timestamp)

    def on_tick(self, timestamp: float) -> None:
        """Wall-clock watchdog: end an active visit even when no frames arrive.

        Without this, a total camera outage during a visit leaves the FSM stuck
        in ``active`` forever, since exit is only evaluated on frame arrival.
        Only advances time forward and never refreshes per-camera qualification.
        """
        if self.state != "active":
            return
        if self._last_ts is not None and timestamp <= self._last_ts:
            return
        self._last_ts = timestamp
        self._maybe_end_visit(timestamp)

    def pause(self, timestamp: float) -> None:
        if self.state == "active":
            self._end_visit(timestamp, enter_cooldown=False)
        self.state = "idle"
        self.enter_accumulator = 0.0
        self.cooldown_until = None
        self._last_frame_qualified = dict.fromkeys(self.camera_ids, False)

    def _start_visit(self, timestamp: float) -> None:
        self.state = "active"
        self.visit_id = str(uuid.uuid4())
        self.started_at = timestamp
        self.enter_accumulator = 0.0
        if self.on_visit_start and self.visit_id:
            self.on_visit_start(self.visit_id)

    def _maybe_end_visit(self, timestamp: float) -> None:
        if not all(
            timestamp - self._last_qualified_at[cam] > self.exit_no_cat_sec
            for cam in self.camera_ids
        ):
            return
        self._end_visit(timestamp, enter_cooldown=True)

    def _end_visit(self, timestamp: float, *, enter_cooldown: bool) -> None:
        if self.visit_id is None or self.started_at is None:
            return

        duration = max(0.0, timestamp - self.started_at)
        discard = duration < self.min_visit_sec

        if self.on_visit_end:
            self.on_visit_end(
                self.visit_id,
                self.started_at,
                timestamp,
                duration,
                discard,
            )

        self.visit_id = None
        self.started_at = None

        if enter_cooldown:
            self.state = "cooldown"
            self.cooldown_until = timestamp + self.cooldown_sec
        else:
            self.state = "idle"

        self.enter_accumulator = 0.0
