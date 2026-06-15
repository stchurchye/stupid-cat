"""Visit video recorder (spec §9.4)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def visit_recording_filename(visit_id: str, camera_id: str, *, primary_camera: str) -> str:
    if camera_id == primary_camera:
        return f"{visit_id}.mp4"
    return f"{visit_id}_{camera_id}.mp4"

_BROWSER_CODECS = ("avc1", "H264", "mp4v")


class VisitRecorder:
    """Write visit clips to data/recordings/ (capped at max_seconds per camera)."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        max_seconds: int = 30,
        fps: float = 6.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_seconds = max_seconds
        self.fps = fps
        self._max_frames = max(1, int(max_seconds * fps))
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._fourcc = "avc1"
        self._frames_written = 0
        self._start_mono: float | None = None

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def start(self, visit_id: str, frame_bgr: np.ndarray, *, filename: str | None = None) -> Path:
        self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / (filename or f"{visit_id}.mp4")
        h, w = frame_bgr.shape[:2]
        self._writer, self._fourcc = _open_video_writer(self._path, self.fps, (w, h))
        self._frames_written = 0
        self._start_mono = time.monotonic()
        self.write_frame(frame_bgr)
        return self._path

    def write_frame(self, frame_bgr: np.ndarray) -> bool:
        if self._writer is None:
            return False
        if self._frames_written >= self._max_frames:
            return False
        # Wall-clock cap: a near-still cat throttles to idle fps, so a fixed frame
        # budget could span minutes; stop once real elapsed time exceeds the limit.
        if self._start_mono is not None and time.monotonic() - self._start_mono > self.max_seconds:
            return False
        self._writer.write(frame_bgr)
        self._frames_written += 1
        return True

    def stop(self, *, reencode: bool = True) -> Path | None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        path = self._path
        self._path = None
        if reencode and path is not None and self._fourcc == "mp4v":
            # ffmpeg re-mux can take seconds and stop() runs inside the visit-end
            # path that the pipeline holds a lock around — do it on a daemon thread
            # so the 24/7 loop is never blocked. Skipped when the caller will
            # discard the clip (reencode=False) to avoid racing the unlink.
            threading.Thread(
                target=reencode_for_browser,
                args=(path,),
                name=f"reencode-{path.stem}",
                daemon=True,
            ).start()
        return path


def _open_video_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, str]:
    w, h = size
    for codec in _BROWSER_CODECS:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (w, h),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"cannot open video writer: {path}")


def _replace_with_retry(tmp: Path, path: Path, *, attempts: int = 5, delay: float = 0.5) -> bool:
    """os.replace with retry: on Windows the target raises PermissionError while
    it's open (e.g. being served/played), which is usually transient."""
    for i in range(attempts):
        try:
            tmp.replace(path)
            return True
        except OSError as exc:
            if i == attempts - 1:
                logger.warning(
                    "could not replace %s after reencode (%d tries): %s", path, attempts, exc
                )
                tmp.unlink(missing_ok=True)
                return False
            time.sleep(delay)
    return False


def reencode_for_browser(path: Path) -> bool:
    """Re-mux to H.264 so Safari/Chrome can play in <video> (needs ffmpeg)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found; %s may not play in browser (mp4v)", path)
        return False

    tmp = path.with_suffix(".browser.mp4")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "ffmpeg reencode failed for %s: %s", path, exc.stderr.decode(errors="replace")[:200]
        )
        tmp.unlink(missing_ok=True)
        return False
    return _replace_with_retry(tmp, path)
