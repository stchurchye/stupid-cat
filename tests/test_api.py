from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stupid_cat.api.app import create_app
from stupid_cat.config import load_config
from stupid_cat.db import Database
from stupid_cat.pipeline import Pipeline


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    db = Database(tmp_path / "test.db")
    pipeline = Pipeline(cfg, db=db, data_dir=tmp_path / "data")
    app = create_app(pipeline, db)
    return TestClient(app)


@pytest.fixture
def db(client: TestClient, tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


def test_health(client: TestClient) -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert "paused" in body
    assert body["db_ok"] is True
    assert body["status"] == "ok"
    assert body["ingest_active"] is False


def test_correct_invalid_cat_returns_400(client: TestClient, db: Database) -> None:
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.0,
    )
    res = client.post(f"/api/v1/visits/{vid}/correct", json={"cat_id": "not_a_real_cat"})
    assert res.status_code == 400


def test_recordings_served_from_pipeline_data_dir(tmp_path: Path) -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    db = Database(tmp_path / "test.db")
    pipeline = Pipeline(cfg, db=db, data_dir=tmp_path / "data")
    rec_dir = pipeline.recordings_dir
    rec_dir.mkdir(parents=True)
    vid = "abc-123"
    (rec_dir / f"{vid}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    app = create_app(pipeline, db)
    client = TestClient(app)
    res = client.get(f"/recordings/{vid}.mp4")
    assert res.status_code == 200


def test_correct_visit_delegates_to_pipeline(client: TestClient, db: Database) -> None:
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.0,
    )
    res = client.post(f"/api/v1/visits/{vid}/correct", json={"cat_id": "mimi"})
    assert res.status_code == 200
    assert db.get_visit(vid)["cat_id"] == "mimi"


def test_cats_and_visits(client: TestClient, db: Database) -> None:
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="mimi",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.8,
    )

    cats = client.get("/api/v1/cats").json()
    assert len(cats) >= 1

    visits = client.get("/api/v1/visits").json()
    assert len(visits) == 1
    assert visits[0]["cat_id"] == "mimi"
    assert visits[0]["recordings"] == []


def test_visit_recordings_dual_files(tmp_path: Path) -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    cfg.recorder.record_cameras = ["cam1", "cam2"]
    db = Database(tmp_path / "test2.db")
    pipeline = Pipeline(cfg, db=db, data_dir=tmp_path / "data")
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.0,
        recording_path=f"data/recordings/{vid}.mp4",
    )
    rec_dir = pipeline.recordings_dir
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / f"{vid}.mp4").write_bytes(b"\x00")
    (rec_dir / f"{vid}_cam2.mp4").write_bytes(b"\x00")

    client = TestClient(create_app(pipeline, db))
    visits = client.get("/api/v1/visits").json()
    assert len(visits) == 1
    recs = visits[0]["recordings"]
    assert len(recs) == 2
    urls = {r["camera_id"]: r["url"] for r in recs}
    assert urls["cam1"] == f"/recordings/{vid}.mp4"
    assert urls["cam2"] == f"/recordings/{vid}_cam2.mp4"


def test_list_cameras(client: TestClient) -> None:
    res = client.get("/api/v1/cameras")
    assert res.status_code == 200
    body = res.json()
    ids = [c["id"] for c in body]
    assert "cam1" in ids
    assert all("name" in c for c in body)


def test_stats_api(client: TestClient, db: Database) -> None:
    db.seed_cats([{"id": "mimi", "name": "咪咪"}])
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="mimi",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=90,
        confidence=0.7,
    )

    res = client.get("/api/v1/stats?days=0")
    assert res.status_code == 200
    body = res.json()
    assert body["total_visits"] == 1
    assert body["total_duration_sec"] == 90
    assert body["by_cat"][0]["cat_id"] == "mimi"


def test_camera_preview_missing_frame(client: TestClient) -> None:
    res = client.get("/api/v1/cameras/cam1/preview.jpg")
    assert res.status_code == 404


def test_camera_preview_returns_jpeg(tmp_path: Path) -> None:
    import numpy as np

    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    db = Database(tmp_path / "test.db")
    import time as _time

    pipeline = Pipeline(cfg, db=db, data_dir=tmp_path / "data")
    with pipeline._preview_lock:
        pipeline._preview_frames["cam1"] = np.zeros((120, 160, 3), dtype=np.uint8)
        pipeline._preview_ts["cam1"] = _time.monotonic()  # fresh, else staleness-gated

    app = create_app(pipeline, db)
    client = TestClient(app)
    res = client.get("/api/v1/cameras/cam1/preview.jpg")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content[:2] == b"\xff\xd8"
