"""Main vision pipeline (spec §4)."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from stupid_cat.config import AppConfig, load_config
from stupid_cat.db import Database
from stupid_cat.detector import CatDetector, select_primary_bbox
from stupid_cat.geometry import bbox_roi_overlap_ratio
from stupid_cat.ingest import FrameEvent, FrameSource, MultiCameraIngest, VideoFileSource
from stupid_cat.preprocess import preprocess_frame
from stupid_cat.recorder import VisitRecorder, reencode_for_browser, visit_recording_filename
from stupid_cat.reid import (
    EmbeddingBuffer,
    EmbeddingRecord,
    Embedder,
    build_centroid_from_refs,
    fuse_embeddings,
    load_all_centroids,
    match_cat,
    ref_quality_ok,
)
from stupid_cat.session import VisitSessionFSM

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _duration_seconds_wall(started_at: str, ended_at: str) -> int:
    """Wall-clock visit duration per spec §8.2.5."""
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return max(0, int((end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()))


def _crop_bgr(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2].copy()


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        db: Database | None = None,
        db_path: Path | str | None = None,
        data_dir: Path | str | None = None,
        detector: CatDetector | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.cfg = cfg
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ingest_active = False
        self._ingest: MultiCameraIngest | None = None
        # Guards all state shared between the pipeline thread and API threads:
        # the FSM, centroids, embedding buffer, recorder, and per-visit fields.
        self._lock = threading.RLock()

        self._data_dir = Path(data_dir or "data")
        self.db = db or Database(db_path or self._data_dir / "stupid_cat.db")
        self.db.init_schema()
        self.db.seed_cats([{"id": c.id, "name": c.name} for c in cfg.cats.seed])

        self.detector = detector or CatDetector(cfg.inference)
        self.embedder = embedder or Embedder(
            device=cfg.inference.device,
            backbone=cfg.inference.reid_backbone,
            fp16=cfg.inference.fp16,
        )
        self.centroids = load_all_centroids(self._data_dir / "cats")
        # Only enabled cameras drive ingest / the FSM exit set.
        self._cameras = [c for c in cfg.cameras if c.enabled]
        if not self._cameras:
            raise ValueError("no enabled cameras")
        self.camera_weights = {c.id: c.weight for c in self._cameras}
        self.roi_by_camera = {c.id: c.roi_polygon for c in self._cameras}
        self._stream_dims = {c.id: (c.stream_width, c.stream_height) for c in self._cameras}
        # Cache of ROI polygons scaled to the actual decoded frame size, keyed by
        # camera id -> (frame_w, frame_h, scaled_polygon). Lets a lower-res
        # substream reuse a main-stream-calibrated ROI (spec §7.3).
        self._roi_scaled: dict[str, tuple[int, int, list]] = {}

        self.fsm = VisitSessionFSM(
            camera_ids=[c.id for c in self._cameras],
            enter_overlap_sec=cfg.session.enter_overlap_sec,
            exit_no_cat_sec=cfg.session.exit_no_cat_sec,
            cooldown_sec=cfg.session.cooldown_sec,
            min_visit_sec=cfg.session.min_visit_sec,
        )
        self.recorder = VisitRecorder(
            self._data_dir / "recordings",
            max_seconds=cfg.recorder.max_seconds,
            fps=float(cfg.inference.active_fps),
        )

        self._embedding_buffer: EmbeddingBuffer | None = None
        self._visit_recorders: dict[str, VisitRecorder] = {}
        self._recording_path: Path | None = None
        self._camera_ids_seen: set[str] = set()
        self._visit_started_at: str | None = None
        self._best_correction_crops: dict[str, np.ndarray] = {}
        self._best_correction_areas: dict[str, float] = {}
        self._last_frame_at: dict[str, str] = {}
        self._preview_lock = threading.Lock()
        self._preview_frames: dict[str, np.ndarray] = {}
        self._preview_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
        self._preview_qualified: dict[str, bool] = {}
        self._preview_ts: dict[str, float] = {}
        self._record_this_visit = False
        self._last_disk_warn_mono = 0.0
        self._consecutive_frame_errors = 0
        # Set while a visit is active so ingest records at full fps (not throttled
        # to idle by a still cat). Wall-clock time of the last qualified detection,
        # used as the visit's true end so a larger exit_no_cat_sec doesn't inflate
        # the reported duration.
        self._visit_active = threading.Event()
        self._last_qualified_iso: str | None = None
        self._max_cats_seen = 0  # most cats simultaneously in-ROI during the visit

        self.fsm.on_visit_start = self._on_visit_start
        self.fsm.on_visit_end = self._on_visit_end

        self._recover_orphan_visits()

    def _recover_orphan_visits(self) -> None:
        """Finalize visits left open by a previous crash and relink any clips."""
        try:
            ids = self.db.finalize_orphan_visits()
        except Exception:  # noqa: BLE001 - recovery must never block startup
            logger.exception("orphan visit recovery failed")
            return
        clips_to_reencode: list[Path] = []
        for visit_id in ids:
            # Match the primary '{id}.mp4' AND per-camera '{id}_{cam}.mp4' clips
            # so a crash-interrupted dual-cam visit relinks/re-encodes every clip.
            clips = sorted(self.recordings_dir.glob(f"{visit_id}*.mp4"))
            if not clips:
                continue
            primary_clip = self.recordings_dir / f"{visit_id}.mp4"
            link = primary_clip if primary_clip.exists() else clips[0]
            try:
                self.db.set_recording_path(visit_id, str(link))
            except Exception:  # noqa: BLE001
                logger.exception("could not relink recording for %s", visit_id)
            clips_to_reencode.extend(clips)
        # Re-encode recovered clips off the startup-critical path so ingest can
        # begin immediately instead of blocking on serial ffmpeg transcodes.
        if clips_to_reencode:
            threading.Thread(
                target=self._reencode_clips,
                args=(clips_to_reencode,),
                name="recover-reencode",
                daemon=True,
            ).start()
        if ids:
            logger.warning(
                "recovered %d interrupted visit(s) from a previous run: %s",
                len(ids),
                ids,
            )

    @staticmethod
    def _reencode_clips(clips: list[Path]) -> None:
        for clip in clips:
            try:
                reencode_for_browser(clip)
            except Exception:  # noqa: BLE001
                logger.warning("could not re-encode recovered clip %s", clip)

    @property
    def ingest_active(self) -> bool:
        return self._ingest_active

    @property
    def frame_error_streak(self) -> int:
        """Consecutive _process_frame failures; a high value means a persistent
        fault (bad model/GPU) is degrading the pipeline into a no-op loop."""
        return self._consecutive_frame_errors

    @property
    def recordings_dir(self) -> Path:
        return self._data_dir / "recordings"

    def record_camera_ids(self) -> list[str]:
        configured = self.cfg.recorder.record_cameras
        if configured is not None:
            return [c for c in configured if c in self.camera_weights]
        return [self.cfg.recorder.primary_camera]

    def _stop_visit_recorders(self, *, reencode: bool = True) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for camera_id, rec in list(self._visit_recorders.items()):
            path = rec.stop(reencode=reencode)
            if path is not None:
                paths[camera_id] = path
        self._visit_recorders.clear()
        return paths

    def _write_visit_recording(self, camera_id: str, visit_id: str, frame: np.ndarray) -> None:
        rec = self._visit_recorders.get(camera_id)
        if rec is None:
            rec = VisitRecorder(
                self.recordings_dir,
                max_seconds=self.cfg.recorder.max_seconds,
                fps=float(self.cfg.inference.active_fps),
            )
            filename = visit_recording_filename(
                visit_id,
                camera_id,
                primary_camera=self.cfg.recorder.primary_camera,
            )
            rec.start(visit_id, frame, filename=filename)
            self._visit_recorders[camera_id] = rec
            if camera_id == self.cfg.recorder.primary_camera:
                self._recording_path = self.recordings_dir / filename
        else:
            rec.write_frame(frame)

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def active_visit_id(self) -> str | None:
        """Current visit id, read under the lock (FSM is mutated on another thread)."""
        with self._lock:
            return self.fsm.visit_id

    def camera_health(self) -> list[dict[str, object]]:
        """Per-camera status for GET /health (spec §10)."""
        with self._lock:
            last_frame_at = dict(self._last_frame_at)
        rows: list[dict[str, object]] = []
        for cam in self._cameras:
            last_iso = last_frame_at.get(cam.id)
            connected = False
            if last_iso is not None:
                try:
                    last_dt = datetime.fromisoformat(last_iso)
                    age = (datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)).total_seconds()
                    connected = age < 30.0
                except ValueError:
                    connected = True
            rows.append(
                {
                    "id": cam.id,
                    "name": cam.name or cam.id,
                    "connected": connected,
                    "last_frame_at": last_iso,
                    "fps_actual": None,
                }
            )
        return rows

    def camera_ids(self) -> list[str]:
        return [c.id for c in self.cfg.cameras]

    def get_preview_jpeg(
        self, camera_id: str, *, quality: int = 82, max_age_sec: float = 10.0
    ) -> bytes | None:
        with self._preview_lock:
            frame = self._preview_frames.get(camera_id)
            boxes = list(self._preview_boxes.get(camera_id, []))
            qualified = self._preview_qualified.get(camera_id, False)
            roi = list(self.roi_by_camera.get(camera_id, []))
            ts = self._preview_ts.get(camera_id)

        # Treat a stale frame as "no frame" so the live UI shows a frozen/dead
        # camera as waiting rather than a still image labelled connected.
        if frame is None or ts is None or (_mono_now() - ts) > max_age_sec:
            return None

        annotated = frame.copy()
        if len(roi) >= 3:
            # Scale the calibration-resolution ROI to the actual decoded frame so
            # the drawn box matches where qualification happens (spec §7.3).
            fh, fw = annotated.shape[:2]
            sw, sh = self._stream_dims.get(camera_id, (fw, fh))
            if sw > 0 and sh > 0 and (sw != fw or sh != fh):
                roi = [[p[0] * fw / sw, p[1] * fh / sh] for p in roi]
            pts = np.array(roi, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        for bbox in boxes:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            color = (0, 200, 255) if qualified else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if qualified:
            cv2.putText(
                annotated,
                "in ROI",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )

        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def pause(self) -> None:
        """Pause ingest and end any active visit (spec §8.2.7)."""
        self._paused.set()
        with self._lock:
            self.fsm.pause(_mono_now())
            # Don't let a pre-pause error streak keep /health "degraded" while
            # the system is intentionally (and healthily) paused.
            self._consecutive_frame_errors = 0

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._ingest is not None:
            self._ingest.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def start_background(self, sources: list[FrameSource]) -> None:
        self._thread = threading.Thread(
            target=self.run,
            args=(sources,),
            name="stupid-cat-pipeline",
            daemon=True,
        )
        self._thread.start()

    def run(self, sources: list[FrameSource]) -> None:
        ingest = MultiCameraIngest(
            sources,
            active_fps=float(self.cfg.inference.active_fps),
            idle_fps=float(self.cfg.inference.idle_fps),
            motion_threshold=float(self.cfg.inference.motion_threshold),
            is_active=self._visit_active.is_set,
        )
        self._ingest = ingest
        self._ingest_active = True
        try:
            for event in ingest.events():
                if self._stop.is_set():
                    break
                if event is None:
                    # Heartbeat during an outage: let the FSM time out an active
                    # visit even though no frames are arriving.
                    if not self._paused.is_set():
                        try:
                            with self._lock:
                                self.fsm.on_tick(_mono_now())
                        except Exception:  # noqa: BLE001 - watchdog must not kill ingest
                            logger.exception("error in watchdog tick; skipping")
                    continue
                if self._paused.is_set():
                    continue
                try:
                    if self._process_frame(event):
                        self._consecutive_frame_errors = 0  # reset only on real work
                except Exception:  # noqa: BLE001 - one bad frame must not kill 24/7 ingest
                    self._consecutive_frame_errors += 1
                    logger.exception(
                        "error processing frame from %s; skipping (streak=%d)",
                        event.camera_id,
                        self._consecutive_frame_errors,
                    )
        finally:
            ingest.stop()
            self._ingest = None
            self._ingest_active = False

    def _disk_ok(self) -> bool:
        """True if free disk space is above recorder.min_free_mb (throttled warn)."""
        min_free = self.cfg.recorder.min_free_mb * 1024 * 1024
        if min_free <= 0:
            return True
        try:
            free = shutil.disk_usage(self._data_dir).free
        except OSError:
            return True  # can't tell — don't block recording
        if free >= min_free:
            return True
        now = _mono_now()
        if now - self._last_disk_warn_mono > 60.0:
            self._last_disk_warn_mono = now
            logger.warning(
                "low disk space (%.0f MB free < %d MB); skipping recording",
                free / 1024 / 1024,
                self.cfg.recorder.min_free_mb,
            )
        return False

    def _correction_crop_path(self, visit_id: str, camera_id: str) -> Path:
        return self._data_dir / "correction_crops" / f"{visit_id}_{camera_id}.jpg"

    def _legacy_correction_crop_path(self, visit_id: str) -> Path:
        return self._data_dir / "correction_crops" / f"{visit_id}.jpg"

    def _save_correction_crops(self, visit_id: str, crops: dict[str, np.ndarray]) -> None:
        if not crops:
            return
        out_dir = self._data_dir / "correction_crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        for camera_id, crop in crops.items():
            path = self._correction_crop_path(visit_id, camera_id)
            if not cv2.imwrite(str(path), crop):
                logger.warning("failed to write correction crop %s", path)

    def _load_correction_crops(self, visit_id: str) -> dict[str, np.ndarray]:
        crops: dict[str, np.ndarray] = {}
        prefix = f"{visit_id}_"
        crop_dir = self._data_dir / "correction_crops"
        if crop_dir.is_dir():
            for path in sorted(crop_dir.glob(f"{visit_id}_*.jpg")):
                if not path.name.startswith(prefix):
                    continue
                camera_id = path.name[len(prefix) :].removesuffix(".jpg")
                if not camera_id:
                    continue
                crop = cv2.imread(str(path))
                if crop is not None:
                    crops[camera_id] = crop
        if not crops:
            legacy = self._legacy_correction_crop_path(visit_id)
            if legacy.exists():
                crop = cv2.imread(str(legacy))
                if crop is not None:
                    crops[self.cfg.recorder.primary_camera] = crop
        return crops

    def _delete_correction_crops(self, visit_id: str) -> None:
        crop_dir = self._data_dir / "correction_crops"
        if not crop_dir.is_dir():
            return
        for path in crop_dir.glob(f"{visit_id}*.jpg"):
            path.unlink(missing_ok=True)

    def _roi_for_frame(self, camera_id: str, frame_w: int, frame_h: int) -> list:
        """ROI polygon scaled from its calibration resolution to the actual frame.

        ROIs are authored in the camera's ``stream_width``×``stream_height``; if the
        decoded frame differs (e.g. a lower-res substream), scale so the overlap
        test stays in one coordinate space (spec §7.3).
        """
        roi = self.roi_by_camera.get(camera_id, [])
        if not roi or frame_w <= 0 or frame_h <= 0:
            return roi
        sw, sh = self._stream_dims.get(camera_id, (frame_w, frame_h))
        if sw <= 0 or sh <= 0 or (sw == frame_w and sh == frame_h):
            return roi
        cached = self._roi_scaled.get(camera_id)
        if cached is not None and cached[0] == frame_w and cached[1] == frame_h:
            return cached[2]
        sx = frame_w / sw
        sy = frame_h / sh
        scaled = [[float(p[0]) * sx, float(p[1]) * sy] for p in roi]
        self._roi_scaled[camera_id] = (frame_w, frame_h, scaled)
        return scaled

    def _process_frame(self, event: FrameEvent) -> bool:
        """Process one frame. Returns True only if it did Re-ID work (embedded a
        crop), so the caller resets the error streak only on real work — an idle
        frame neither resets nor masks a persistent embed fault."""
        camera_id = event.camera_id

        frame = preprocess_frame(event.frame_bgr, self.cfg.preprocess)
        boxes = self.detector.detect(frame, camera_id)
        primary = select_primary_bbox(boxes)
        qualified = False
        in_roi_count = 0  # how many cats are simultaneously inside the ROI
        if primary is not None:
            h, w = frame.shape[:2]
            roi = self._roi_for_frame(camera_id, w, h)
            thr = self.cfg.session.roi_overlap_min
            qualified = bbox_roi_overlap_ratio(primary, roi) >= thr  # largest box (spec §8.4)
            in_roi_count = sum(1 for b in boxes if bbox_roi_overlap_ratio(b, roi) >= thr)

        with self._lock:
            now_iso = _iso_now()
            self._last_frame_at[camera_id] = now_iso
            if qualified:
                self._last_qualified_iso = now_iso  # true presence end (any camera)
            self.fsm.on_frame(camera_id, qualified=qualified, timestamp=event.timestamp)
            visit_active = self.fsm.state == "active"
            visit_id = self.fsm.visit_id
            if visit_active and in_roi_count > self._max_cats_seen:
                self._max_cats_seen = in_roi_count  # flag multi-cat visits
            embed_work = visit_active and primary is not None

        with self._preview_lock:
            self._preview_frames[camera_id] = frame.copy()
            self._preview_boxes[camera_id] = boxes
            self._preview_qualified[camera_id] = qualified
            self._preview_ts[camera_id] = _mono_now()

        if visit_active:
            with self._lock:
                # Re-read LIVE fsm state under the lock: pause()/visit-end may have
                # run since the first lock block was released. Writing to a cleared
                # recorder would recreate (and truncate) the just-finalized clip.
                if self.fsm.state == "active" and self.fsm.visit_id == visit_id:
                    self._camera_ids_seen.add(camera_id)
                    if (
                        self._record_this_visit
                        and camera_id in self.record_camera_ids()
                        and visit_id
                    ):
                        self._write_visit_recording(camera_id, visit_id, frame)

        if not embed_work:
            return False

        with self._lock:

            crop = _crop_bgr(frame, primary)
            short_side = min(crop.shape[0], crop.shape[1])
            if short_side < self.cfg.inference.min_crop_px or self._embedding_buffer is None:
                return False
            # True bbox area (width * height) so buffer eviction keeps the
            # largest crops as spec §8.6 intends (was short_side**2 before).
            bbox_area = max(0.0, primary[2] - primary[0]) * max(0.0, primary[3] - primary[1])
            weight = self.camera_weights.get(camera_id, 0.5)
            visit_id = self.fsm.visit_id
            best_area = self._best_correction_areas.get(camera_id, -1.0)

        # Re-ID forward pass is the slow part — run it WITHOUT the lock so the API
        # (/health) and the FSM watchdog aren't blocked for the inference time.
        # `crop` is a private copy and the Embedder is internally locked.
        embedding = self.embedder.embed(crop)
        # Only the larger-than-best crop can become the correction representative,
        # so skip the (heavier) percentile quality check otherwise.
        crop_ok = bbox_area > best_area and ref_quality_ok(crop)

        with self._lock:
            # The visit may have ended (e.g. via pause) while we embedded; only
            # attach to the still-current visit's buffer.
            if self._embedding_buffer is None or self.fsm.visit_id != visit_id:
                return False
            # Per camera: keep the largest, clearest crop for correction refs.
            if bbox_area > self._best_correction_areas.get(camera_id, -1.0) and crop_ok:
                self._best_correction_areas[camera_id] = bbox_area
                self._best_correction_crops[camera_id] = crop
            self._embedding_buffer.add(
                EmbeddingRecord(
                    embedding=embedding,
                    weight=weight,
                    bbox_area=bbox_area,
                    timestamp=event.timestamp,
                )
            )
            return True  # did real Re-ID work this frame

    def _on_visit_start(self, visit_id: str) -> None:
        # Defensively close any writer left open by an abnormally-ended visit so
        # we never leak a handle or keep writing into the wrong file.
        self._stop_visit_recorders(reencode=False)
        self.recorder.stop(reencode=False)
        self._embedding_buffer = EmbeddingBuffer(self.cfg.inference.fusion_max_frames)
        self._camera_ids_seen = set()
        self._recording_path = None
        self._visit_started_at = _iso_now()
        self._best_correction_crops = {}
        self._best_correction_areas = {}
        self._last_qualified_iso = None
        self._max_cats_seen = 0
        self._visit_active.set()  # ingest now records at full fps until visit end
        # Decide once per visit whether to record (config + disk space).
        self._record_this_visit = self.cfg.recorder.enabled and self._disk_ok()
        # Persist the visit at START so a crash mid-visit leaves a recoverable
        # row instead of losing it entirely (finalized on next startup).
        try:
            self.db.create_visit(
                visit_id=visit_id,
                cat_id="unknown",
                started_at=self._visit_started_at,
                camera_ids=[],
            )
        except Exception:  # noqa: BLE001 - never let a DB hiccup kill the FSM
            logger.exception("failed to persist visit start %s", visit_id)
        logger.info("visit started %s", visit_id)

    def _on_visit_end(
        self,
        visit_id: str,
        _started_mono: float,
        _ended_mono: float,
        duration: float,
        discard: bool,
    ) -> None:
        self._visit_active.clear()  # back to motion-gated idle fps
        # Skip the (background) browser re-encode when we're about to discard the
        # clip, so it can't race the unlink below.
        recording_paths = self._stop_visit_recorders(reencode=not discard)
        self.recorder.stop(reencode=not discard)
        recording_path = recording_paths.get(self.cfg.recorder.primary_camera)
        if recording_path is None and recording_paths:
            # Primary never qualified but a secondary cam recorded — still store a
            # clip path so the DB row / crash-recovery has something to point at.
            recording_path = next(iter(recording_paths.values()))
        self._recording_path = None

        if discard:
            for path in recording_paths.values():
                if path.exists():
                    path.unlink(missing_ok=True)
            self._embedding_buffer = None
            self._best_correction_crops = {}
            self._best_correction_areas = {}
            self._delete_correction_crops(visit_id)
            # Row was created at visit start; drop it since the visit is discarded.
            try:
                self.db.delete_visit(visit_id)
            except Exception:  # noqa: BLE001
                logger.exception("failed to delete discarded visit %s", visit_id)
            logger.info("visit %s discarded (duration %.2fs)", visit_id, duration)
            return

        if self._best_correction_crops:
            # Save only the single best (largest) crop for the visit: one
            # appearance = one ref, so a dual-camera visit isn't double-counted in
            # the unweighted centroid (and one correction can't satisfy min_refs).
            best_cam = max(
                self._best_correction_crops,
                key=lambda cam: self._best_correction_areas.get(cam, -1.0),
            )
            self._save_correction_crops(
                visit_id, {best_cam: self._best_correction_crops[best_cam]}
            )
        self._best_correction_crops = {}
        self._best_correction_areas = {}

        embs, weights = [], []
        if self._embedding_buffer and len(self._embedding_buffer) > 0:
            embs, weights = self._embedding_buffer.embeddings_and_weights()

        if embs:
            visit_vector = fuse_embeddings(
                embs,
                weights,
                self.cfg.inference.fusion,
                centroids=self.centroids or None,
            )
            cat_id, confidence = match_cat(
                visit_vector,
                self.centroids,
                self.cfg.inference.similarity_threshold,
            )
            frames_used = len(self._embedding_buffer)
        else:
            cat_id, confidence = "unknown", 0.0
            frames_used = 0

        started_at = self._visit_started_at or _iso_now()
        # End the visit at the LAST time a cat was actually seen (any camera), not
        # the no-cat-timeout moment, so widening exit_no_cat_sec to avoid splitting
        # a turning cat doesn't inflate the reported duration.
        ended_at = self._last_qualified_iso or _iso_now()
        duration_sec = _duration_seconds_wall(started_at, ended_at)
        camera_ids = sorted(self._camera_ids_seen)
        max_cats = max(1, self._max_cats_seen)

        def _finalize() -> None:
            self.db.end_visit(
                visit_id,
                cat_id=cat_id,
                ended_at=ended_at,
                duration_sec=duration_sec,
                confidence=confidence,
                frames_used=frames_used,
                camera_ids=camera_ids,
                recording_path=str(recording_path) if recording_path else None,
                max_cats=max_cats,
            )

        # The row was created at visit start; if it is somehow missing (start
        # write failed), recreate it so the visit is not lost.
        try:
            _finalize()
        except KeyError:
            self.db.create_visit(
                visit_id=visit_id,
                cat_id="unknown",
                started_at=started_at,
                camera_ids=camera_ids,
            )
            _finalize()

        self._visit_started_at = None
        self._embedding_buffer = None
        logger.info(
            "visit ended %s cat=%s conf=%.2f max_cats=%d", visit_id, cat_id, confidence, max_cats
        )
        if max_cats >= 2:
            logger.warning(
                "visit %s saw %d cats in the box at once — identity is unreliable",
                visit_id, max_cats,
            )

    def _set_centroid(self, cat_id: str, centroid: np.ndarray) -> None:
        """Replace the centroids dict atomically (copy-on-write).

        Readers in the pipeline thread snapshot ``self.centroids`` and iterate the
        snapshot, so rebinding here can never raise "dict changed size during
        iteration"; the lock additionally serializes concurrent writers.
        """
        self.centroids = {**self.centroids, cat_id: centroid}

    def _collect_visit_ref_crops(self, visit_id: str) -> dict[str, np.ndarray]:
        """Read this visit's already-saved refs (from a prior correction) so a
        re-correction can re-apply them to a different cat."""
        crops: dict[str, np.ndarray] = {}
        cats_dir = self._data_dir / "cats"
        if not cats_dir.is_dir():
            return crops
        prefix = f"{visit_id}_"
        for cat_dir in sorted(cats_dir.iterdir()):
            refs = cat_dir / "refs"
            if not refs.is_dir():
                continue
            for path in sorted(refs.glob(f"{visit_id}_*.jpg")):
                cam = path.name[len(prefix):].removesuffix(".jpg")
                if cam and cam not in crops:
                    img = cv2.imread(str(path))
                    if img is not None:
                        crops[cam] = img
            legacy = refs / f"{visit_id}.jpg"
            if legacy.exists() and self.cfg.recorder.primary_camera not in crops:
                img = cv2.imread(str(legacy))
                if img is not None:
                    crops[self.cfg.recorder.primary_camera] = img
        return crops

    def _delete_visit_refs(self, visit_id: str, *, except_cat: str | None = None) -> set[str]:
        """Delete this visit's refs from every cat except ``except_cat``.

        Returns the ids of cats whose refs changed (so their centroids rebuild)."""
        affected: set[str] = set()
        cats_dir = self._data_dir / "cats"
        if not cats_dir.is_dir():
            return affected
        for cat_dir in cats_dir.iterdir():
            if not cat_dir.is_dir() or cat_dir.name == except_cat:
                continue
            refs = cat_dir / "refs"
            if not refs.is_dir():
                continue
            removed = False
            paths = list(refs.glob(f"{visit_id}_*.jpg")) + [refs / f"{visit_id}.jpg"]
            for path in paths:
                if path.exists():
                    path.unlink(missing_ok=True)
                    removed = True
            if removed:
                affected.add(cat_dir.name)
        return affected

    def _rebuild_centroid_for(self, cat_id: str) -> None:
        """Rebuild (or remove) one cat's centroid from its current refs. Heavy
        build runs off the pipeline lock; only the swap/remove is locked."""
        cat_dir = self._data_dir / "cats" / cat_id
        refs_dir = cat_dir / "refs"
        centroid = None
        if refs_dir.is_dir():
            centroid = build_centroid_from_refs(
                self.embedder, refs_dir, self.cfg.preprocess, self.cfg.cats.min_refs
            )
        centroid_path = cat_dir / "centroid.npy"
        if centroid is None:
            centroid_path.unlink(missing_ok=True)
            with self._lock:
                if cat_id in self.centroids:
                    self.centroids = {k: v for k, v in self.centroids.items() if k != cat_id}
        else:
            np.save(centroid_path, centroid)
            with self._lock:
                self._set_centroid(cat_id, centroid)

    def correct_visit(self, visit_id: str, new_cat_id: str) -> None:
        """DB correction plus ref move and centroid rebuild (spec §9.3).

        A correction re-assigns the WHOLE visit: the visit's representative crop is
        moved to the new cat's refs and removed from any cat it was previously
        assigned to (re-correction must not leave stale refs biasing the old cat).
        Both affected cats' centroids are rebuilt. Correcting to ``unknown`` just
        removes the visit's learning. Heavy rebuilds run off the pipeline lock.
        """
        allowed = {c["id"] for c in self.db.list_cats()} | {"unknown"}
        if new_cat_id not in allowed:
            raise ValueError(f"cat_id not registered: {new_cat_id}")

        self.db.correct_visit(visit_id, new_cat_id)

        # Crop source: the freshly-captured crops (first correction) or, if those
        # were already consumed, the refs saved under the previous cat.
        crops = self._load_correction_crops(visit_id) or self._collect_visit_ref_crops(visit_id)
        rebuild_cats = self._delete_visit_refs(visit_id, except_cat=new_cat_id)

        if new_cat_id != "unknown":
            if crops:
                refs_dir = self._data_dir / "cats" / new_cat_id / "refs"
                refs_dir.mkdir(parents=True, exist_ok=True)
                for camera_id, crop in crops.items():
                    ref_path = refs_dir / f"{visit_id}_{camera_id}.jpg"
                    if cv2.imwrite(str(ref_path), crop):
                        rebuild_cats.add(new_cat_id)
                    else:
                        logger.warning("failed to write ref %s", ref_path)
            else:
                logger.warning("no correction crops for visit %s; DB updated only", visit_id)

        self._delete_correction_crops(visit_id)

        for cat in rebuild_cats:
            self._rebuild_centroid_for(cat)
        if rebuild_cats:
            logger.info("correction rebuilt centroids for %s", sorted(rebuild_cats))

    def rebuild_cat_centroid(self, cat_id: str) -> bool:
        """Rebuild one cat centroid from refs/ (spec §10 rebuild-embedding).

        Centroid build runs off the pipeline lock; only the swap-in is locked.
        """
        refs_dir = self._data_dir / "cats" / cat_id / "refs"
        if not refs_dir.is_dir():
            return False
        centroid = build_centroid_from_refs(
            self.embedder,
            refs_dir,
            self.cfg.preprocess,
            self.cfg.cats.min_refs,
        )
        if centroid is None:
            return False
        cat_dir = refs_dir.parent
        np.save(cat_dir / "centroid.npy", centroid)
        with self._lock:
            self._set_centroid(cat_id, centroid)
        return True


def _mono_now() -> float:
    return time.monotonic()


def build_pipeline(
    config_path: Path | str,
    *,
    local_config_path: Path | str | None = None,
    video_path: Path | str | None = None,
) -> tuple[Pipeline, list[FrameSource]]:
    cfg = load_config(Path(config_path), local_path=local_config_path)
    pipeline = Pipeline(cfg)

    if video_path is not None:
        enabled_ids = [c.id for c in cfg.cameras if c.enabled]
        # Use primary_camera if it's an enabled camera, else the first enabled one,
        # so the FSM (built from enabled cameras) recognizes the source id.
        cam_id = cfg.recorder.primary_camera if cfg.recorder.primary_camera in enabled_ids else enabled_ids[0]
        sources: list[FrameSource] = [VideoFileSource(cam_id, video_path)]
        return pipeline, sources

    from stupid_cat.ingest import RtspSource

    sources = [RtspSource(c.id, c.rtsp_url) for c in cfg.cameras if c.enabled]
    return pipeline, sources
