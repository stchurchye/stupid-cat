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
from stupid_cat.recorder import VisitRecorder, reencode_for_browser
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
        self._recording_path: Path | None = None
        self._camera_ids_seen: set[str] = set()
        self._visit_started_at: str | None = None
        self._best_correction_crop: np.ndarray | None = None
        self._best_correction_area: float = -1.0
        self._last_frame_at: dict[str, str] = {}
        self._record_this_visit = False
        self._last_disk_warn_mono = 0.0
        self._consecutive_frame_errors = 0

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
            clip = self.recordings_dir / f"{visit_id}.mp4"
            if clip.exists():
                try:
                    self.db.set_recording_path(visit_id, str(clip))
                except Exception:  # noqa: BLE001
                    logger.exception("could not relink recording for %s", visit_id)
                clips_to_reencode.append(clip)
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
                    "connected": connected,
                    "last_frame_at": last_iso,
                    "fps_actual": None,
                }
            )
        return rows

    def pause(self) -> None:
        """Pause ingest and end any active visit (spec §8.2.7)."""
        self._paused.set()
        with self._lock:
            self.fsm.pause(_mono_now())

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
                    self._process_frame(event)
                    self._consecutive_frame_errors = 0
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

    def _correction_crop_path(self, visit_id: str) -> Path:
        return self._data_dir / "correction_crops" / f"{visit_id}.jpg"

    def _save_correction_crop(self, visit_id: str, crop: np.ndarray) -> None:
        path = self._correction_crop_path(visit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop):
            logger.warning("failed to write correction crop %s", path)

    def _load_correction_crop(self, visit_id: str) -> np.ndarray | None:
        path = self._correction_crop_path(visit_id)
        if not path.exists():
            return None
        crop = cv2.imread(str(path))
        return crop

    def _delete_correction_crop(self, visit_id: str) -> None:
        self._correction_crop_path(visit_id).unlink(missing_ok=True)

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

    def _process_frame(self, event: FrameEvent) -> None:
        camera_id = event.camera_id

        frame = preprocess_frame(event.frame_bgr, self.cfg.preprocess)
        boxes = self.detector.detect(frame, camera_id)
        primary = select_primary_bbox(boxes)
        qualified = False
        if primary is not None:
            h, w = frame.shape[:2]
            roi = self._roi_for_frame(camera_id, w, h)
            overlap = bbox_roi_overlap_ratio(primary, roi)
            qualified = overlap >= self.cfg.session.roi_overlap_min

        with self._lock:
            self._last_frame_at[camera_id] = _iso_now()
            self.fsm.on_frame(camera_id, qualified=qualified, timestamp=event.timestamp)

            if self.fsm.state != "active" or primary is None:
                return

            self._camera_ids_seen.add(camera_id)
            if (
                self._record_this_visit
                and camera_id == self.cfg.recorder.primary_camera
                and self.fsm.visit_id
            ):
                if self._recording_path is None:
                    self._recording_path = self.recorder.start(self.fsm.visit_id, frame)
                else:
                    self.recorder.write_frame(frame)

            crop = _crop_bgr(frame, primary)
            short_side = min(crop.shape[0], crop.shape[1])
            if short_side < self.cfg.inference.min_crop_px or self._embedding_buffer is None:
                return
            # True bbox area (width * height) so buffer eviction keeps the
            # largest crops as spec §8.6 intends (was short_side**2 before).
            bbox_area = max(0.0, primary[2] - primary[0]) * max(0.0, primary[3] - primary[1])
            weight = self.camera_weights.get(camera_id, 0.5)
            visit_id = self.fsm.visit_id

        # Re-ID forward pass is the slow part — run it WITHOUT the lock so the API
        # (/health) and the FSM watchdog aren't blocked for the inference time.
        # `crop` is a private copy and the Embedder is internally locked.
        embedding = self.embedder.embed(crop)
        crop_ok = ref_quality_ok(crop)

        with self._lock:
            # The visit may have ended (e.g. via pause) while we embedded; only
            # attach to the still-current visit's buffer.
            if self._embedding_buffer is None or self.fsm.visit_id != visit_id:
                return
            # Keep the largest, clearest crop as the visit's representative for the
            # correction gallery — chosen by crop size, NOT by similarity to an
            # existing centroid (which would bias corrections toward whatever the
            # model already guessed). Must also pass the ref-quality gate, else the
            # saved ref would be dropped at centroid-build time (correction no-op).
            if bbox_area > self._best_correction_area and crop_ok:
                self._best_correction_area = bbox_area
                self._best_correction_crop = crop
            self._embedding_buffer.add(
                EmbeddingRecord(
                    embedding=embedding,
                    weight=weight,
                    bbox_area=bbox_area,
                    timestamp=event.timestamp,
                )
            )

    def _on_visit_start(self, visit_id: str) -> None:
        # Defensively close any writer left open by an abnormally-ended visit so
        # we never leak a handle or keep writing into the wrong file.
        self.recorder.stop()
        self._embedding_buffer = EmbeddingBuffer(self.cfg.inference.fusion_max_frames)
        self._camera_ids_seen = set()
        self._recording_path = None
        self._visit_started_at = _iso_now()
        self._best_correction_crop = None
        self._best_correction_area = -1.0
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
        # Skip the (background) browser re-encode when we're about to discard the
        # clip, so it can't race the unlink below.
        recording_path = self.recorder.stop(reencode=not discard)
        self._recording_path = None

        if discard:
            if recording_path and recording_path.exists():
                recording_path.unlink(missing_ok=True)
            self._embedding_buffer = None
            self._best_correction_crop = None
            self._best_correction_area = -1.0
            self._delete_correction_crop(visit_id)
            # Row was created at visit start; drop it since the visit is discarded.
            try:
                self.db.delete_visit(visit_id)
            except Exception:  # noqa: BLE001
                logger.exception("failed to delete discarded visit %s", visit_id)
            logger.info("visit %s discarded (duration %.2fs)", visit_id, duration)
            return

        if self._best_correction_crop is not None:
            self._save_correction_crop(visit_id, self._best_correction_crop)
        self._best_correction_crop = None
        self._best_correction_area = -1.0

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
        ended_at = _iso_now()
        duration_sec = _duration_seconds_wall(started_at, ended_at)
        camera_ids = sorted(self._camera_ids_seen)

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
        logger.info("visit ended %s cat=%s conf=%.2f", visit_id, cat_id, confidence)

    def _set_centroid(self, cat_id: str, centroid: np.ndarray) -> None:
        """Replace the centroids dict atomically (copy-on-write).

        Readers in the pipeline thread snapshot ``self.centroids`` and iterate the
        snapshot, so rebinding here can never raise "dict changed size during
        iteration"; the lock additionally serializes concurrent writers.
        """
        self.centroids = {**self.centroids, cat_id: centroid}

    def correct_visit(self, visit_id: str, new_cat_id: str) -> None:
        """DB correction plus optional ref append and centroid rebuild (spec §9.3).

        The expensive centroid rebuild (a Re-ID forward pass per ref image) runs
        WITHOUT the pipeline lock so an admin correction can't freeze frame
        ingest / the FSM watchdog; only the cheap copy-on-write swap is locked.
        The DB and Embedder each have their own internal locks.
        """
        allowed = {c["id"] for c in self.db.list_cats()} | {"unknown"}
        if new_cat_id not in allowed:
            raise ValueError(f"cat_id not registered: {new_cat_id}")

        self.db.correct_visit(visit_id, new_cat_id)
        if new_cat_id == "unknown":
            self._delete_correction_crop(visit_id)
            return

        crop = self._load_correction_crop(visit_id)
        if crop is None:
            logger.warning("no correction crop for visit %s; DB updated only", visit_id)
            return

        cat_dir = self._data_dir / "cats" / new_cat_id
        refs_dir = cat_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        ref_path = refs_dir / f"{visit_id}.jpg"
        if not cv2.imwrite(str(ref_path), crop):
            logger.warning("failed to write ref %s", ref_path)
            return

        self._delete_correction_crop(visit_id)

        centroid = build_centroid_from_refs(
            self.embedder,
            refs_dir,
            self.cfg.preprocess,
            self.cfg.cats.min_refs,
        )
        if centroid is None:
            logger.info(
                "ref saved for %s; centroid unchanged (need %d refs)",
                new_cat_id,
                self.cfg.cats.min_refs,
            )
            return

        np.save(cat_dir / "centroid.npy", centroid)
        with self._lock:
            self._set_centroid(new_cat_id, centroid)
        logger.info("rebuilt centroid for %s", new_cat_id)

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
        cam_id = cfg.recorder.primary_camera
        sources: list[FrameSource] = [VideoFileSource(cam_id, video_path)]
        return pipeline, sources

    from stupid_cat.ingest import RtspSource

    sources = [RtspSource(c.id, c.rtsp_url) for c in cfg.cameras if c.enabled]
    return pipeline, sources
