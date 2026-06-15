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


def test_visit_stats(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.seed_cats([{"id": "mimi", "name": "咪咪"}])
    v1 = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        v1,
        cat_id="mimi",
        ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120,
        confidence=0.8,
    )
    v2 = db.create_visit(cat_id="unknown", started_at="2026-06-02T18:00:00+08:00")
    db.end_visit(
        v2,
        cat_id="unknown",
        ended_at="2026-06-02T18:01:00+08:00",
        duration_sec=60,
        confidence=0.0,
    )

    stats = db.visit_stats(from_ts="2026-06-02T00:00:00+08:00")
    assert stats["total_visits"] == 2
    assert stats["total_duration_sec"] == 180
    assert len(stats["by_day"]) == 1
    assert stats["by_day"][0]["visits"] == 2

    mimi = next(c for c in stats["by_cat"] if c["cat_id"] == "mimi")
    assert mimi["name"] == "咪咪"
    assert mimi["visits"] == 1
    assert mimi["duration_sec"] == 120


def test_visit_stats_respects_timezone_cutoff(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-07T23:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-07T23:01:00+08:00",
        duration_sec=60,
        confidence=0.0,
    )

    included = db.visit_stats(from_ts="2026-06-07T22:00:00+08:00")
    assert included["total_visits"] == 1

    excluded = db.visit_stats(from_ts="2026-06-08T00:00:00+08:00")
    assert excluded["total_visits"] == 0


def test_clear_recording_paths(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid,
        cat_id="unknown",
        ended_at="2026-06-02T10:01:00+08:00",
        duration_sec=60,
        confidence=0.0,
        recording_path=f"data/recordings/{vid}.mp4",
    )
    db.clear_recording_paths([vid])
    assert db.get_visit(vid)["recording_path"] is None



def test_end_visit_stores_digging_motion(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid, cat_id="mimi", ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120, confidence=0.5, digging_motion=42.5,
    )
    assert db.get_visit(vid)["digging_motion"] == 42.5


def test_set_waste_logs_correction_and_accuracy(tmp_path: Path) -> None:
    db = _db(tmp_path)
    vid = db.create_visit(cat_id="unknown", started_at="2026-06-02T10:00:00+08:00")
    db.end_visit(
        vid, cat_id="mimi", ended_at="2026-06-02T10:02:00+08:00",
        duration_sec=120, confidence=0.5, waste_type="pee", waste_confidence=0.6,
    )
    db.set_waste_type(vid, "poop")  # heuristic said pee, human says poop
    acc = db.waste_accuracy()
    assert acc["total_corrections"] == 1
    assert acc["confusion"]["pee->poop"] == 1
    db.set_waste_type(vid, "poop")  # no change -> not logged again
    assert db.waste_accuracy()["total_corrections"] == 1
