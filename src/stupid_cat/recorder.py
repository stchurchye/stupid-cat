"""Visit video recorder (spec §9.4)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BROWSER_CODECS = ("avc1", "H264", "mp4v")


class VisitRecorder:
    """Write primary-camera frames to data/recordings/{visit_id}.mp4 (capped at max_seconds)."""

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

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def start(self, visit_id: str, frame_bgr: np.ndarray) -> Path:
        self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / f"{visit_id}.mp4"
        h, w = frame_bgr.shape[:2]
        self._writer, self._fourcc = _open_video_writer(self._path, self.fps, (w, h))
        self._frames_written = 0
        self.write_frame(frame_bgr)
        return self._path

    def write_frame(self, frame_bgr: np.ndarray) -> bool:
        if self._writer is None:
            return False
        if self._frames_written >= self._max_frames:
            return False
        self._writer.write(frame_bgr)
        self._frames_written += 1
        return True

    def stop(self) -> Path | None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        path = self._path
        self._path = None
        if path is not None and self._fourcc == "mp4v":
            reencode_for_browser(path)
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
        tmp.replace(path)
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg reencode failed for %s: %s", path, exc.stderr.decode(errors="replace")[:200])
        tmp.unlink(missing_ok=True)
        return False
