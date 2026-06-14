"""Frame sources, rate limiting, and motion gating (spec §4.1)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sentinel pushed by a reader thread when its source is exhausted.
_SENTINEL = object()


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
    """Merge per-camera streams, each read on its own thread.

    Each camera runs in a dedicated daemon thread feeding a bounded queue, so a
    stall or RTSP reconnect on one camera never blocks the other (spec §4.1/§4.2).
    When the queue fills, the oldest frame is dropped to bound end-to-end latency.
    ``events()`` yields ``None`` as a heartbeat when no frame arrives within
    ``poll_timeout`` so the consumer can drive wall-clock timeouts during a total
    outage.
    """

    def __init__(
        self,
        sources: list[FrameSource],
        *,
        active_fps: float,
        idle_fps: float,
        motion_threshold: float,
        queue_maxsize: int = 8,
        poll_timeout: float = 0.5,
    ) -> None:
        if not sources:
            raise ValueError("sources must not be empty")
        self._sources = list(sources)
        self._active_fps = active_fps
        self._idle_fps = idle_fps
        self._motion_threshold = motion_threshold
        self._poll_timeout = poll_timeout
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _put_drop_oldest(self, event: FrameEvent) -> None:
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()  # drop the oldest frame
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    def _reader(self, source: FrameSource) -> None:
        try:
            for event in rate_limited(
                source,
                active_fps=self._active_fps,
                idle_fps=self._idle_fps,
                motion_threshold=self._motion_threshold,
            ):
                if self._stop.is_set():
                    break
                self._put_drop_oldest(event)
        except Exception:  # noqa: BLE001 - one camera must not kill the others
            logger.exception("ingest reader for %s crashed", getattr(source, "camera_id", "?"))
        finally:
            self._queue.put(_SENTINEL)

    def events(self) -> Generator[FrameEvent | None, None, None]:
        self._threads = [
            threading.Thread(
                target=self._reader,
                args=(s,),
                name=f"ingest-{s.camera_id}",
                daemon=True,
            )
            for s in self._sources
        ]
        for thread in self._threads:
            thread.start()

        remaining = len(self._threads)
        while remaining > 0 and not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self._poll_timeout)
            except queue.Empty:
                yield None  # heartbeat: no frames arrived (possible total outage)
                continue
            if item is _SENTINEL:
                remaining -= 1
                continue
            yield item

    def stop(self) -> None:
        self._stop.set()
