from pathlib import Path

import pytest

from stupid_cat.db import Database, VisitAlreadyEndedError


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.init_schema()
    return db


def test_insert_visit(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.seed_cats([{"id": "mimi", "name": "咪咪"}])
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.0,
    )
    row = db.get_visit(vid)
    assert row is not None
    assert row["duration_sec"] == 120
    assert row["cat_id"] == "unknown"
    assert row["waste_type"] == "unknown"


def test_end_visit_sets_final_cat_id_and_frames_used(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="mimi",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.82,
        frames_used=17,
    )
    row = db.get_visit(vid)
    assert row["cat_id"] == "mimi"
    assert row["frames_used"] == 17
    assert row["ended_at"] is not None


def test_end_visit_missing_raises(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with pytest.raises(KeyError, match="visit not found"):
        db.end_visit(
            "missing",
            cat_id="unknown",
            ended_at="2026-06-02T10:01:00+08:00",
            duration_sec=60,
            confidence=0.0,
        )


def test_end_visit_twice_raises(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="mimi",
        ended_at="2026-06-02T10:01:00+08:00",
        duration_sec=60,
        confidence=0.5,
    )
    with pytest.raises(VisitAlreadyEndedError):
        db.end_visit(
            vid,
            cat_id="mimi",
            ended_at="2026-06-02T10:05:00+08:00",
            duration_sec=300,
            confidence=0.9,
        )


def test_list_visits_filters_by_time_and_cat(tmp_path: Path) -> None:
    db = _db(tmp_path)
    early = db.create_visit(cat_id="unknown", started_at="2026-06-01T10:00:00+08:00")
    db.end_visit(
        early,
        cat_id="unknown",
        ended_at="2026-06-01T10:01:00+08:00",
        duration_sec=60,
        confidence=0.0,
    )

    mimi_id = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        mimi_id,
        cat_id="mimi",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.82,
        frames_used=10,
        camera_ids=["cam1", "cam2"],
    )

    by_cat = db.list_visits(cat_id="mimi")
    assert len(by_cat) == 1
    assert by_cat[0]["id"] == mimi_id
    assert by_cat[0]["camera_ids"] == ["cam1", "cam2"]

    in_range = db.list_visits(
        from_ts="2026-06-02T00:00:00+08:00",
        to_ts="2026-06-02T23:59:59+08:00",
    )
    assert len(in_range) == 1
    assert in_range[0]["cat_id"] == "mimi"


def test_correct_visit_updates_cat_and_records_correction(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.seed_cats([{"id": "mimi", "name": "咪咪"}, {"id": "cat2", "name": "猫2"}])
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:01:00+08:00",
        duration_sec=60,
        confidence=0.4,
    )

    db.correct_visit(vid, "mimi")
    row = db.get_visit(vid)
    assert row is not None
    assert row["cat_id"] == "mimi"
    assert row["corrected"] is True

    corrections = db.list_corrections(visit_id=vid)
    assert len(corrections) == 1
    assert corrections[0]["old_cat_id"] == "unknown"
    assert corrections[0]["new_cat_id"] == "mimi"


def test_correct_visit_unknown_target_allowed(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="mimi",
        ended_at="2026-06-02T10:01:00+08:00",
        duration_sec=60,
        confidence=0.9,
    )
    db.correct_visit(vid, "unknown")
    assert db.get_visit(vid)["cat_id"] == "unknown"


def test_get_visit_missing_returns_none(tmp_path: Path) -> None:
    db = _db(tmp_path)
    assert db.get_visit("nonexistent") is None


def test_correct_visit_missing_raises(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with pytest.raises(KeyError):
        db.correct_visit("missing", "mimi")


def test_database_context_manager_closes_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "test.db"
    with Database(path) as db:
        db.init_schema()
        vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
        db.end_visit(
            vid,
            cat_id="mimi",
            ended_at="2026-06-02T10:01:00+08:00",
            duration_sec=60,
            confidence=0.5,
        )
    assert Database(path)._conn is None

    reopened = Database(path)
    row = reopened.get_visit(vid)
    assert row is not None
    assert row["cat_id"] == "mimi"
    reopened.close()
