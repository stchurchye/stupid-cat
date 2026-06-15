"""FastAPI application (spec §10)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from stupid_cat.db import Database
from stupid_cat.pipeline import Pipeline
from stupid_cat.recorder import visit_recording_filename
from stupid_cat.timeutil import local_iso_cutoff

# Reachable without the API key so health probes work and the login page can load.
_AUTH_PUBLIC_PATHS = frozenset({"/api/v1/health", "/login", "/api/v1/login"})
_AUTH_COOKIE = "sc_key"

_LOGIN_HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>stupid-cat 登录</title>
<style>body{font-family:system-ui;display:grid;place-items:center;height:100vh;margin:0;background:#0f172a;color:#e2e8f0}
form{background:#1e293b;padding:24px;border-radius:10px;display:flex;flex-direction:column;gap:12px;min-width:260px}
input,button{padding:10px;border-radius:6px;border:1px solid #334155;font-size:1em}
button{background:#2563eb;color:#fff;border:none;cursor:pointer}#err{color:#f87171;font-size:.9em;min-height:1em}</style>
</head><body><form id="f"><h2>🐱 需要访问密钥</h2>
<input id="k" type="password" placeholder="API key" autofocus autocomplete="current-password">
<div id="err"></div><button type="submit">进入</button></form>
<script>document.getElementById('f').addEventListener('submit',async e=>{e.preventDefault();
const r=await fetch('/api/v1/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({key:document.getElementById('k').value})});
if(r.ok){location.href='/';}else{document.getElementById('err').textContent='密钥错误';}});</script>
</body></html>"""


class CorrectVisitBody(BaseModel):
    cat_id: str


class WasteBody(BaseModel):
    waste_type: str


class LoginBody(BaseModel):
    key: str


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Gate every request (except health/login) behind a shared key supplied via
    the X-API-Key header or the sc_key cookie. A browser hitting an HTML page
    without the cookie is redirected to /login; programmatic clients get 401."""

    def __init__(self, app: object, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if path in _AUTH_PUBLIC_PATHS:
            return await call_next(request)
        supplied = request.headers.get("X-API-Key") or request.cookies.get(_AUTH_COOKIE)
        if supplied != self._api_key:
            if "text/html" in request.headers.get("accept", ""):
                return RedirectResponse("/login")
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)


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
    svc = pipeline.cfg.service

    # Reject spoofed Host headers (DNS-rebinding) unless explicitly disabled.
    if svc.trusted_hosts and svc.trusted_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=svc.trusted_hosts)
    # CORS only when origins are configured; default is same-origin only.
    if svc.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=svc.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # Optional shared-secret gate (no-op when api_key is empty).
    if svc.api_key:
        app.add_middleware(ApiKeyMiddleware, api_key=svc.api_key)

    @app.get("/login", response_class=HTMLResponse)
    def login_page() -> str:
        return _LOGIN_HTML

    @app.post("/api/v1/login")
    def login(body: LoginBody) -> Response:
        if not svc.api_key or body.key != svc.api_key:
            raise HTTPException(status_code=401, detail="invalid key")
        resp = JSONResponse({"ok": True})
        # httponly so page JS can't leak it; sent automatically on same-origin
        # fetch()/video requests, so the bundled UI works once logged in.
        resp.set_cookie(
            _AUTH_COOKIE, svc.api_key, httponly=True, samesite="lax", max_age=30 * 86400
        )
        return resp

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

    @app.get("/api/v1/waste/accuracy")
    def waste_accuracy() -> dict:
        """Predicted-vs-corrected pee/poop summary for tuning the heuristic."""
        return db.waste_accuracy()

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
