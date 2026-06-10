"""Frame sources, rate limiting, and motion gating (spec §4.1)."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameEvent:
    camera_id: str
    frame_bgr: np.ndarray
    timestamp: float


class FrameSource(Protocol):
    camera_id: str

    def frames(self) -> Iterator[np.ndarray]: ...


class VideoFileSource:
    def __init__(self, camera_id: str, path: Path | str) -> None:
        self.camera_id = camera_id
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def frames(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


class RtspSource:
    """RTSP reader with reconnect/backoff when the stream drops."""

    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        reconnect_delay_sec: float = 5.0,
    ) -> None:
        self.camera_id = camera_id
        self.url = url
        self.reconnect_delay_sec = reconnect_delay_sec

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.warning("RTSP open failed for %s, retry in %.0fs", self.camera_id, self.reconnect_delay_sec)
                time.sleep(self.reconnect_delay_sec)
                continue
            logger.info("RTSP connected: %s", self.camera_id)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        logger.warning("RTSP read ended for %s, reconnecting", self.camera_id)
                        break
                    yield frame
            finally:
                cap.release()
            time.sleep(self.reconnect_delay_sec)


def motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def rate_limited(
    source: FrameSource,
    *,
    active_fps: float,
    idle_fps: float,
    motion_threshold: float,
) -> Generator[FrameEvent, None, None]:
    """Yield frames with motion-aware fps throttling."""
    prev_gray: np.ndarray | None = None
    last_emit = 0.0
    active = False

    for frame in source.frames():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = 0.0 if prev_gray is None else motion_score(prev_gray, gray)
        prev_gray = gray
        active = score >= motion_threshold
        target_fps = active_fps if active else idle_fps
        interval = 1.0 / max(target_fps, 0.1)

        now = time.monotonic()
        if now - last_emit < interval:
            continue
        last_emit = now
        yield FrameEvent(camera_id=source.camera_id, frame_bgr=frame, timestamp=now)


class MultiCameraIngest:
    """Round-robin merge of per-camera rate-limited streams."""

    def __init__(self, sources: list[FrameSource], *, active_fps: float, idle_fps: float, motion_threshold: float) -> None:
        if not sources:
            raise ValueError("sources must not be empty")
        self._generators = [
            rate_limited(s, active_fps=active_fps, idle_fps=idle_fps, motion_threshold=motion_threshold)
            for s in sources
        ]
        self._indices = list(range(len(self._generators)))

    def events(self) -> Generator[FrameEvent, None, None]:
        active = set(self._indices)
        while active:
            for idx in list(active):
                gen = self._generators[idx]
                try:
                    yield next(gen)
                except StopIteration:
                    active.discard(idx)
