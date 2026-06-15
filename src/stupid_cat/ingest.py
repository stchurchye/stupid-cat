"""Frame sources, rate limiting, and motion gating (spec §4.1)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Generator, Iterator
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
        max_reconnect_delay_sec: float = 300.0,
    ) -> None:
        self.camera_id = camera_id
        self.url = url
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_delay_sec = max_reconnect_delay_sec
        self._warned_no_timeout = False

    def _open(self) -> cv2.VideoCapture:
        # OPEN/READ timeouts are open-only properties: they must be passed in the
        # constructor params, NOT set() after opening (which is a no-op). This
        # bounds the RTSP handshake so a half-dead host can't block the reader.
        params: list[int] = []
        for prop_name, value in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000),
            ("CAP_PROP_READ_TIMEOUT_MSEC", 5000),
        ):
            prop = getattr(cv2, prop_name, None)
            if prop is not None:
                params += [int(prop), value]
        if params:
            try:
                return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            except Exception:  # noqa: BLE001 - any builds rejecting the 3-arg form fall back
                # Surface the fallback once: without timeout params a dead host can
                # block open/read far longer, which on a shared executor can starve
                # the sibling camera. Visible logging beats a silent hang.
                if not self._warned_no_timeout:
                    logger.warning(
                        "%s: OpenCV rejected RTSP open/read timeout params; falling "
                        "back to no-timeout open (a dead host may block longer)",
                        self.camera_id,
                    )
                    self._warned_no_timeout = True
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _backoff_delay(self, attempt: int) -> float:
        return min(self.reconnect_delay_sec * (2 ** attempt), self.max_reconnect_delay_sec)

    def frames(self) -> Iterator[np.ndarray]:
        # Exponential backoff (base, 2x, ... capped at max) so an hour-long outage
        # isn't an hour of busy 5s retries; a stream that connects and delivers a
        # frame resets the backoff so a brief blip reconnects fast.
        attempt = 0
        while True:
            cap = self._open()
            if not cap.isOpened():
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "RTSP open failed for %s, retry in %.0fs (attempt %d)",
                    self.camera_id, delay, attempt + 1,
                )
                time.sleep(delay)
                attempt = min(attempt + 1, 16)
                continue
            # Keep only the freshest frame (process live video, not a backlog).
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
            logger.info("RTSP connected: %s", self.camera_id)
            got_frame = False
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        logger.warning("RTSP read ended for %s, reconnecting", self.camera_id)
                        break
                    if not got_frame:
                        got_frame = True
                        attempt = 0  # healthy stream -> reset backoff
                    yield frame
            finally:
                cap.release()
            delay = self._backoff_delay(attempt)
            time.sleep(delay)
            attempt = min(attempt + 1, 16)


def motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def _motion_gray(frame: np.ndarray) -> np.ndarray:
    """Downsampled grayscale for motion gating — cheap on full-res IR frames."""
    h, w = frame.shape[:2]
    if w > 320:
        scale = 320.0 / w
        frame = cv2.resize(
            frame, (320, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def rate_limited(
    source: FrameSource,
    *,
    active_fps: float,
    idle_fps: float,
    motion_threshold: float,
    is_active: "Callable[[], bool] | None" = None,
) -> Generator[FrameEvent, None, None]:
    """Yield frames with motion-aware fps throttling.

    ``is_active`` (optional) is polled per frame: while it returns True (a visit
    is in progress) frames are emitted at ``active_fps`` regardless of motion, so
    the recording stays complete and plays at real time even when the cat sits
    still (which would otherwise throttle to ``idle_fps``).
    """
    prev_gray: np.ndarray | None = None
    last_emit = 0.0
    active = False

    for frame in source.frames():
        gray = _motion_gray(frame)
        # A reconnect can renegotiate a different resolution/aspect ratio, so the
        # gray shape may change mid-stream; absdiff would raise on a mismatch.
        # Treat the first frame and any shape change as idle (no motion) and
        # re-baseline — set active directly so this holds even if motion_threshold
        # is 0 (where score >= 0 would otherwise read as active).
        if prev_gray is None or prev_gray.shape != gray.shape:
            active = False
        else:
            active = motion_score(prev_gray, gray) >= motion_threshold
        prev_gray = gray
        if is_active is not None and is_active():
            active = True  # full fps during a visit, even for a still cat
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
        is_active: "Callable[[], bool] | None" = None,
    ) -> None:
        if not sources:
            raise ValueError("sources must not be empty")
        self._sources = list(sources)
        self._active_fps = active_fps
        self._idle_fps = idle_fps
        self._motion_threshold = motion_threshold
        self._poll_timeout = poll_timeout
        self._is_active = is_active
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # Count of readers that have exited. Tracked out-of-band (NOT via a
        # sentinel in the bounded drop-oldest queue) so a dropped item can never
        # lose a termination signal — which would otherwise hang events() for
        # 3+ cameras under backpressure.
        self._finished = 0
        self._finished_lock = threading.Lock()

    def _put_drop_oldest(self, event: FrameEvent) -> None:
        # The queue carries frames only; dropping the oldest can never lose a
        # termination signal because readers signal completion via _finished.
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()  # make room by dropping the oldest frame
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
                is_active=self._is_active,
            ):
                if self._stop.is_set():
                    break
                self._put_drop_oldest(event)
        except Exception:  # noqa: BLE001 - one camera must not kill the others
            logger.exception("ingest reader for %s crashed", getattr(source, "camera_id", "?"))
        finally:
            with self._finished_lock:
                self._finished += 1

    def _all_finished(self) -> bool:
        with self._finished_lock:
            return self._finished >= len(self._threads)

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

        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self._poll_timeout)
            except queue.Empty:
                # Terminate once every reader has exited and the queue is drained;
                # otherwise emit a heartbeat so the consumer can drive timeouts.
                if self._all_finished() and self._queue.empty():
                    break
                yield None
                continue
            yield item

    def stop(self) -> None:
        self._stop.set()
