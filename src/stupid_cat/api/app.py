"""FastAPI application (spec §10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stupid_cat.db import Database
from stupid_cat.pipeline import Pipeline
from stupid_cat.recorder import visit_recording_filename
from stupid_cat.timeutil import local_iso_cutoff


class CorrectVisitBody(BaseModel):
    cat_id: str


class WasteBody(BaseModel):
    waste_type: str


def _visit_recordings(pipeline: Pipeline, visit: dict) -> list[dict[str, str]]:
    visit_id = visit["id"]
    primary = pipeline.cfg.recorder.primary_camera
    name_by_id = {c.id: c.name or c.id for c in pipeline.cfg.cameras}
    out: list[dict[str, str]] = []
    for cam_id in pipeline.record_camera_ids():
        filename = visit_recording_filename(visit_id, cam_id, primary_camera=primary)
        if (pipeline.recordings_dir / filename).exists():
            out.append(
                {
                    "camera_id": cam_id,
                    "name": name_by_id.get(cam_id, cam_id),
                    "url": f"/recordings/{filename}",
                }
            )
    return out


def _enrich_visit(pipeline: Pipeline, visit: dict) -> dict:
    recordings = _visit_recordings(pipeline, visit)
    visit = dict(visit)
    visit["recordings"] = recordings
    if recordings:
        visit["recording_url"] = recordings[0]["url"]
    elif visit.get("recording_path"):
        visit["recording_url"] = f"/recordings/{visit['id']}.mp4"
    return visit


def create_app(pipeline: Pipeline, db: Database) -> FastAPI:
    app = FastAPI(title="stupid-cat")
    static_dir = Path(__file__).resolve().parent.parent / "web" / "static"
    recordings_dir = pipeline.recordings_dir

    @app.get("/api/v1/health")
    def health() -> dict:
        cameras = pipeline.camera_health()
        if pipeline.ingest_active:
            any_disconnected = any(not c["connected"] for c in cameras)
            # A sustained run of frame errors (bad model/GPU) means the pipeline
            # is alive but producing nothing — surface that as degraded.
            stuck = pipeline.frame_error_streak >= 30
            status = "degraded" if (any_disconnected or stuck) else "ok"
        else:
            status = "ok"
        return {
            "status": status,
            "paused": pipeline.paused,
            "device": pipeline.cfg.inference.device,
            "cuda_available": _cuda_available(),
            "cameras": cameras,
            "active_visit_id": pipeline.active_visit_id,
            "db_ok": True,
            "ingest_active": pipeline.ingest_active,
        }

    @app.get("/api/v1/cats")
    def list_cats() -> list:
        return db.list_cats()

    @app.get("/api/v1/visits")
    def list_visits(
        from_ts: str | None = None,
        to_ts: str | None = None,
        cat_id: str | None = None,
        limit: int = 1000,
    ) -> list:
        # Bound the result (and the per-visit recording-file stat()s) so the
        # timeline stays responsive after months of 24/7 visits.
        limit = max(1, min(limit, 5000))
        visits = db.list_visits(
            from_ts=from_ts, to_ts=to_ts, cat_id=cat_id, only_ended=True, limit=limit
        )
        return [_enrich_visit(pipeline, v) for v in visits]

    @app.get("/api/v1/stats")
    def visit_stats(
        days: int | None = 7,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> dict:
        if days is not None and days < 0:
            raise HTTPException(status_code=400, detail="days must be >= 0")
        if from_ts is None and days is not None and days > 0:
            from_ts = local_iso_cutoff(days=days)
        return db.visit_stats(from_ts=from_ts, to_ts=to_ts)

    @app.get("/api/v1/visits/{visit_id}")
    def get_visit(visit_id: str) -> dict:
        row = db.get_visit(visit_id)
        if row is None:
            raise HTTPException(status_code=404, detail="visit not found")
        return _enrich_visit(pipeline, row)

    @app.post("/api/v1/visits/{visit_id}/correct")
    def correct_visit(visit_id: str, body: CorrectVisitBody) -> dict:
        try:
            pipeline.correct_visit(visit_id, body.cat_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "visit_id": visit_id, "cat_id": body.cat_id}

    @app.post("/api/v1/visits/{visit_id}/waste")
    def set_waste(visit_id: str, body: WasteBody) -> dict:
        if body.waste_type not in ("pee", "poop", "unknown"):
            raise HTTPException(status_code=400, detail="waste_type must be pee/poop/unknown")
        try:
            db.set_waste_type(visit_id, body.waste_type)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True, "visit_id": visit_id, "waste_type": body.waste_type}

    @app.post("/api/v1/cats/{cat_id}/rebuild-embedding")
    def rebuild_embedding(cat_id: str) -> dict:
        if not pipeline.rebuild_cat_centroid(cat_id):
            raise HTTPException(
                status_code=400,
                detail=f"cannot rebuild centroid for {cat_id} (missing refs or below min_refs)",
            )
        return {"ok": True, "cat_id": cat_id}

    @app.get("/api/v1/cameras")
    def list_cameras() -> list[dict[str, str]]:
        return [
            {"id": cam.id, "name": cam.name or cam.id}
            for cam in pipeline.cfg.cameras
        ]

    @app.get("/api/v1/cameras/{camera_id}/preview.jpg")
    def camera_preview(camera_id: str) -> Response:
        if camera_id not in pipeline.camera_ids():
            raise HTTPException(status_code=404, detail="camera not found")
        jpeg = pipeline.get_preview_jpeg(camera_id)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.post("/api/v1/pause")
    def pause() -> dict:
        pipeline.pause()
        return {"paused": True}

    @app.post("/api/v1/resume")
    def resume() -> dict:
        pipeline.resume()
        return {"paused": False}

    # Create the dir up front so the mount always succeeds; otherwise on a fresh
    # install it doesn't exist yet at startup and every /recordings/* URL 404s
    # until the process is restarted after the first visit.
    recordings_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/recordings", StaticFiles(directory=recordings_dir), name="recordings")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
