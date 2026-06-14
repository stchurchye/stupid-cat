"""Clear visit history and media; keep cat registry and refs."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

KEEP_CAT_IDS = {"juanjuan", "guagua", "shoulang", "xuemei", "leilei"}
OLD_CAT_DIRS = {"mimi", "cat2", "cat3", "cat4", "cat5"}


def _unlink_files(folder: Path, pattern: str) -> int:
    if not folder.is_dir():
        return 0
    count = 0
    for path in folder.glob(pattern):
        path.unlink(missing_ok=True)
        count += 1
    return count


def clear_data(data_dir: Path, db_path: Path, *, wipe_models: bool) -> None:
    conn = sqlite3.connect(db_path)
    visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    conn.execute("DELETE FROM corrections")
    conn.execute("DELETE FROM visits")
    conn.commit()
    conn.close()

    recordings = _unlink_files(data_dir / "recordings", "*.mp4")
    crops = _unlink_files(data_dir / "correction_crops", "*.jpg")

    cats_dir = data_dir / "cats"
    removed_dirs = 0
    wiped_centroids = 0
    if cats_dir.is_dir():
        for cat_dir in cats_dir.iterdir():
            if cat_dir.is_dir() and cat_dir.name in OLD_CAT_DIRS:
                for path in sorted(cat_dir.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink(missing_ok=True)
                for sub in sorted(cat_dir.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                cat_dir.rmdir()
                removed_dirs += 1
            elif cat_dir.is_dir() and cat_dir.name in KEEP_CAT_IDS:
                refs = cat_dir / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                if wipe_models:
                    centroid = cat_dir / "centroid.npy"
                    if centroid.exists():
                        centroid.unlink()
                        wiped_centroids += 1

    print(f"visits removed: {visits}")
    print(f"recordings removed: {recordings}")
    print(f"correction crops removed: {crops}")
    print(f"old cat dirs removed: {removed_dirs}")
    if wipe_models:
        print(f"centroids removed: {wiped_centroids}")
    else:
        print("centroids kept (use --wipe-models to delete)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--db", type=Path, default=Path("data/stupid_cat.db"))
    parser.add_argument(
        "--wipe-models",
        action="store_true",
        help="also delete centroid.npy for registered cats",
    )
    args = parser.parse_args()
    clear_data(args.data_dir, args.db, wipe_models=args.wipe_models)


if __name__ == "__main__":
    main()
