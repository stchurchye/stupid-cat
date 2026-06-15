#!/usr/bin/env python3
"""Back up the SQLite DB (online/WAL-safe) and optionally zip recordings.

Designed to be run from a scheduler (Windows Task Scheduler / cron). See
docs/ops.md for a ready-to-install scheduled task.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from stupid_cat.db import Database


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db", type=Path, default=Path("data/stupid_cat.db"))
    parser.add_argument("--dest", type=Path, default=Path("backups"))
    parser.add_argument(
        "--zip-recordings",
        action="store_true",
        help="also archive data/recordings/ into the backup folder",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=14,
        help="delete backup folders older than the newest N (0 = keep all)",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.dest / stamp
    out.mkdir(parents=True, exist_ok=True)

    db = Database(args.db)
    try:
        db.backup(out / "stupid_cat.db")
    finally:
        db.close()
    print(f"backed up DB -> {out / 'stupid_cat.db'}")

    if args.zip_recordings:
        rec = args.data_dir / "recordings"
        if rec.is_dir():
            archive = shutil.make_archive(str(out / "recordings"), "zip", root_dir=rec)
            print(f"archived recordings -> {archive}")

    if args.keep > 0:
        backups = sorted(
            (p for p in args.dest.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        for stale in backups[args.keep :]:
            shutil.rmtree(stale, ignore_errors=True)
            print(f"pruned old backup {stale.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
