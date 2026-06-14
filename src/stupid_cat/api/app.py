"""FastAPI application (spec §10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stupid_cat.db import Database
from stupid_cat.pipeline import Pipeline
from stupid_cat.timeutil import local_iso_cutoff


class CorrectVisitBody(BaseModel):
    cat_id: str


def create_app(pipeline: Pipeline, db: Database) -> FastAPI:
    app = FastAPI(title="stupid-cat")
    static_dir = Path(__file__).resolve().parent.parent / "web" / "static"
    recordings_dir = pipeline.recordings_dir

    @app.get("/api/v1/health")
    def health() -> dict:
        cameras = pipeline.camera_health()
        if pipeline.ingest_active:
            any_disconnected = any(not c["connected"] for c in cameras)
            status = "degraded" if any_disconnected else "ok"
        else:
            status = "ok"
        return {
            "status": status,
            "paused": pipeline.paused,
            "device": pipeline.cfg.inference.device,
            "cuda_available": _cuda_available(),
            "cameras": cameras,
            "active_visit_id": pipeline.fsm.visit_id,
            "db_ok": True,
            "ingest_active": pipeline.ingest_active,
        }

    @app.get("/api/v1/cats")
    def list_cats() -> list:
        return db.list_cats()

    @app.get("/api/v1/visits")
    def list_visits(from_ts: str | None = None, to_ts: str | None = None, cat_id: str | None = None) -> list:
        return db.list_visits(from_ts=from_ts, to_ts=to_ts, cat_id=cat_id, only_ended=True)

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
        if row.get("recording_path"):
            row["recording_url"] = f"/recordings/{visit_id}.mp4"
        return row

    @app.post("/api/v1/visits/{visit_id}/correct")
    def correct_visit(visit_id: str, body: CorrectVisitBody) -> dict:
        try:
            pipeline.correct_visit(visit_id, body.cat_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "visit_id": visit_id, "cat_id": body.cat_id}

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

    if recordings_dir.exists():
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
