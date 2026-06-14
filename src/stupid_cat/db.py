"""SQLite persistence for cats, visits, and corrections (spec §9)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stupid_cat.timeutil import in_time_range, local_date_key


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VisitAlreadyEndedError(ValueError):
    """Visit was already finalized with end_visit."""


class Database:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None

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
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
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
            """
        )
        conn.commit()

    def seed_cats(self, cats: list[dict[str, str]]) -> None:
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

    def get_visit(self, visit_id: str) -> dict[str, Any] | None:
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

        if cat_id is not None:
            clauses.append("cat_id = ?")
            params.append(cat_id)
        if only_ended:
            clauses.append("ended_at IS NOT NULL")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM visits {where} ORDER BY started_at DESC"
        rows = self._connection().execute(sql, params).fetchall()
        visits = [self._row_to_visit(r) for r in rows]
        if from_ts is not None or to_ts is not None:
            visits = [
                v
                for v in visits
                if in_time_range(v["started_at"], from_ts=from_ts, to_ts=to_ts)
            ]
        return visits

    def correct_visit(self, visit_id: str, new_cat_id: str) -> None:
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
        rows = self._connection().execute(
            "SELECT id, name, created_at FROM cats ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def visit_stats(
        self,
        *,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> dict[str, Any]:
        visits = self.list_visits(from_ts=from_ts, to_ts=to_ts, only_ended=True)
        cat_names = {c["id"]: c["name"] for c in self.list_cats()}

        by_cat_map: dict[str, dict[str, Any]] = {}
        by_day_map: dict[str, dict[str, int]] = {}
        total_duration = 0

        for visit in visits:
            cid = visit["cat_id"]
            duration = int(visit.get("duration_sec") or 0)
            total_duration += duration
            day = local_date_key(visit["started_at"])

            bucket = by_cat_map.setdefault(
                cid,
                {
                    "cat_id": cid,
                    "name": cat_names.get(cid, cid if cid != "unknown" else "未识别"),
                    "visits": 0,
                    "duration_sec": 0,
                    "last_visit_at": None,
                },
            )
            bucket["visits"] += 1
            bucket["duration_sec"] += duration
            if bucket["last_visit_at"] is None or visit["started_at"] > bucket["last_visit_at"]:
                bucket["last_visit_at"] = visit["started_at"]

            day_bucket = by_day_map.setdefault(day, {"date": day, "visits": 0, "duration_sec": 0})
            day_bucket["visits"] += 1
            day_bucket["duration_sec"] += duration

        by_cat = []
        for bucket in by_cat_map.values():
            avg = bucket["duration_sec"] / bucket["visits"] if bucket["visits"] else 0.0
            by_cat.append(
                {
                    **bucket,
                    "avg_duration_sec": round(avg, 1),
                }
            )
        by_cat.sort(key=lambda row: (-row["visits"], row["cat_id"]))

        by_day = sorted(by_day_map.values(), key=lambda row: row["date"])

        return {
            "from_ts": from_ts,
            "to_ts": to_ts,
            "total_visits": len(visits),
            "total_duration_sec": total_duration,
            "by_cat": by_cat,
            "by_day": by_day,
        }

    def clear_recording_paths(self, visit_ids: list[str]) -> None:
        if not visit_ids:
            return
        conn = self._connection()
        placeholders = ",".join("?" * len(visit_ids))
        conn.execute(
            f"UPDATE visits SET recording_path = NULL WHERE id IN ({placeholders})",
            visit_ids,
        )
        conn.commit()

    @staticmethod
    def _row_to_visit(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["camera_ids"] = json.loads(data["camera_ids"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid camera_ids JSON for visit {data['id']}") from exc
        data["corrected"] = bool(data["corrected"])
        return data
