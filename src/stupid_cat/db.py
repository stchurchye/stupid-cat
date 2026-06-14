"""SQLite persistence for cats, visits, and corrections (spec §9)."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VisitAlreadyEndedError(ValueError):
    """Visit was already finalized with end_visit."""


class Database:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None
        # Serializes access to the single shared connection, which is reused from
        # both the pipeline (ingest) thread and the API (uvicorn) worker threads.
        self._lock = threading.RLock()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def init_schema(self) -> None:
        with self._lock:
            conn = self._connection()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cats (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS visits (
                    id TEXT PRIMARY KEY,
                    cat_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_sec INTEGER,
                    confidence REAL NOT NULL DEFAULT 0,
                    waste_type TEXT NOT NULL DEFAULT 'unknown',
                    waste_confidence REAL NOT NULL DEFAULT 0,
                    frames_used INTEGER NOT NULL DEFAULT 0,
                    camera_ids TEXT NOT NULL DEFAULT '[]',
                    recording_path TEXT,
                    corrected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id TEXT NOT NULL REFERENCES visits(id),
                    old_cat_id TEXT NOT NULL,
                    new_cat_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_visits_started_at ON visits(started_at);
                CREATE INDEX IF NOT EXISTS idx_visits_cat_started ON visits(cat_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_corrections_visit ON corrections(visit_id);
                """
            )
            conn.commit()

    def seed_cats(self, cats: list[dict[str, str]]) -> None:
        with self._lock:
            conn = self._connection()
            for cat in cats:
                conn.execute(
                    """
                    INSERT INTO cats (id, name, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (cat["id"], cat["name"], _utc_now_iso()),
                )
            conn.commit()

    def create_visit(
        self,
        *,
        cat_id: str,
        started_at: str,
        visit_id: str | None = None,
        camera_ids: list[str] | None = None,
    ) -> str:
        visit_id = visit_id or str(uuid.uuid4())
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT INTO visits (
                    id, cat_id, started_at, camera_ids, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    visit_id,
                    cat_id,
                    started_at,
                    json.dumps(camera_ids or []),
                    _utc_now_iso(),
                ),
            )
            conn.commit()
            return visit_id

    def end_visit(
        self,
        visit_id: str,
        *,
        cat_id: str,
        ended_at: str,
        duration_sec: int,
        confidence: float,
        frames_used: int = 0,
        camera_ids: list[str] | None = None,
        recording_path: str | None = None,
    ) -> None:
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT ended_at FROM visits WHERE id = ?",
                (visit_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"visit not found: {visit_id}")
            if row["ended_at"] is not None:
                raise VisitAlreadyEndedError(f"visit already ended: {visit_id}")

            updates: dict[str, Any] = {
                "cat_id": cat_id,
                "ended_at": ended_at,
                "duration_sec": duration_sec,
                "confidence": confidence,
                "frames_used": frames_used,
            }
            if camera_ids is not None:
                updates["camera_ids"] = json.dumps(camera_ids)
            if recording_path is not None:
                updates["recording_path"] = recording_path

            set_clause = ", ".join(f"{col} = ?" for col in updates)
            conn.execute(
                f"UPDATE visits SET {set_clause} WHERE id = ?",
                (*updates.values(), visit_id),
            )
            conn.commit()

    def delete_visit(self, visit_id: str) -> None:
        """Remove a visit and its corrections (used to drop discarded visits)."""
        with self._lock:
            conn = self._connection()
            conn.execute("DELETE FROM corrections WHERE visit_id = ?", (visit_id,))
            conn.execute("DELETE FROM visits WHERE id = ?", (visit_id,))
            conn.commit()

    def set_recording_path(self, visit_id: str, recording_path: str) -> None:
        with self._lock:
            conn = self._connection()
            conn.execute(
                "UPDATE visits SET recording_path = ? WHERE id = ?",
                (recording_path, visit_id),
            )
            conn.commit()

    def finalize_orphan_visits(self) -> list[str]:
        """Close visits left open by a crash/power loss (no ended_at).

        Marks each as ended at its start time with duration 0 so it surfaces in
        the timeline (with any partial recording) for manual review instead of
        being silently lost. Returns the finalized visit ids.
        """
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT id, started_at, created_at FROM visits WHERE ended_at IS NULL"
            ).fetchall()
            ids: list[str] = []
            for row in rows:
                ended = row["started_at"] or row["created_at"]
                conn.execute(
                    "UPDATE visits SET ended_at = ?, "
                    "duration_sec = COALESCE(duration_sec, 0) WHERE id = ?",
                    (ended, row["id"]),
                )
                ids.append(row["id"])
            conn.commit()
            return ids

    def get_visit(self, visit_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM visits WHERE id = ?",
                (visit_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_visit(row)

    def list_visits(
        self,
        *,
        from_ts: str | None = None,
        to_ts: str | None = None,
        cat_id: str | None = None,
        only_ended: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if from_ts is not None:
            clauses.append("started_at >= ?")
            params.append(from_ts)
        if to_ts is not None:
            clauses.append("started_at <= ?")
            params.append(to_ts)
        if cat_id is not None:
            clauses.append("cat_id = ?")
            params.append(cat_id)
        if only_ended:
            clauses.append("ended_at IS NOT NULL")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM visits {where} ORDER BY started_at DESC"
        with self._lock:
            rows = self._connection().execute(sql, params).fetchall()
        return [self._row_to_visit(r) for r in rows]

    def correct_visit(self, visit_id: str, new_cat_id: str) -> None:
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT cat_id FROM visits WHERE id = ?",
                (visit_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"visit not found: {visit_id}")

            old_cat_id = row["cat_id"]
            conn.execute(
                """
                UPDATE visits SET cat_id = ?, corrected = 1 WHERE id = ?
                """,
                (new_cat_id, visit_id),
            )
            conn.execute(
                """
                INSERT INTO corrections (visit_id, old_cat_id, new_cat_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (visit_id, old_cat_id, new_cat_id, _utc_now_iso()),
            )
            conn.commit()

    def list_corrections(self, *, visit_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if visit_id is not None:
                rows = self._connection().execute(
                    """
                    SELECT id, visit_id, old_cat_id, new_cat_id, created_at
                    FROM corrections WHERE visit_id = ? ORDER BY id
                    """,
                    (visit_id,),
                ).fetchall()
            else:
                rows = self._connection().execute(
                    """
                    SELECT id, visit_id, old_cat_id, new_cat_id, created_at
                    FROM corrections ORDER BY id
                    """
                ).fetchall()
        return [dict(r) for r in rows]

    def list_cats(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT id, name, created_at FROM cats ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_visit(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["camera_ids"] = json.loads(data["camera_ids"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid camera_ids JSON for visit {data['id']}") from exc
        data["corrected"] = bool(data["corrected"])
        return data
