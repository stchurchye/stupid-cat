"""Delete old visit recordings and clear stale DB paths."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stupid_cat.db import Database


def prune_recordings(data_dir: Path, db_path: Path, *, days: int) -> None:
    rec_dir = data_dir / "recordings"
    if not rec_dir.is_dir():
        print("recordings dir missing")
        return

    cutoff = datetime.now(timezone.utc).astimezone() - timedelta(days=days)
    removed = 0
    freed = 0
    visit_ids: list[str] = []

    for path in rec_dir.glob("*.mp4"):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()
        if mtime < cutoff:
            freed += path.stat().st_size
            visit_ids.append(path.stem)
            path.unlink(missing_ok=True)
            removed += 1

    if visit_ids:
        db = Database(db_path)
        db.clear_recording_paths(visit_ids)
        db.close()

    mb = round(freed / (1024 * 1024), 1)
    print(f"removed {removed} file(s) older than {days} days (~{mb} MB freed)")
    if visit_ids:
        print(f"cleared recording_path for {len(visit_ids)} visit(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db", type=Path, default=Path("data/stupid_cat.db"))
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    prune_recordings(args.data_dir, args.db, days=args.days)


if __name__ == "__main__":
    main()
