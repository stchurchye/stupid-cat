"""ISO timestamp helpers for visit filtering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def local_iso_cutoff(*, days: int) -> str:
    """Cutoff timestamp in local TZ, matching pipeline visit storage."""
    start = datetime.now(timezone.utc).astimezone() - timedelta(days=days)
    return start.isoformat(timespec="seconds")


def in_time_range(
    started_at: str,
    *,
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> bool:
    dt = parse_iso(started_at)
    if from_ts is not None and dt < parse_iso(from_ts):
        return False
    if to_ts is not None and dt > parse_iso(to_ts):
        return False
    return True


def local_date_key(started_at: str) -> str:
    return parse_iso(started_at).date().isoformat()
