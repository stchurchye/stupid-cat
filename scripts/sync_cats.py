"""Sync five cats into the local database."""
from __future__ import annotations

from pathlib import Path

from stupid_cat.db import Database

CATS = [
    ("juanjuan", "卷卷"),
    ("guagua", "瓜瓜"),
    ("shoulang", "寿郎"),
    ("xuemei", "雪梅"),
    ("leilei", "雷雷"),
]
OLD = ("mimi", "cat2", "cat3", "cat4", "cat5")


def main() -> None:
    for cat_id, _ in CATS:
        (Path("data/cats") / cat_id / "refs").mkdir(parents=True, exist_ok=True)

    db = Database(Path("data/stupid_cat.db"))
    db.init_schema()
    conn = db._connection()
    conn.executemany(
        "INSERT INTO cats (id, name, created_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(id) DO UPDATE SET name=excluded.name",
        CATS,
    )
    conn.execute(
        f"DELETE FROM cats WHERE id IN ({','.join('?' * len(OLD))})",
        OLD,
    )
    conn.commit()
    for row in conn.execute("SELECT id, name FROM cats ORDER BY id"):
        print(f"{row['id']}\t{row['name']}")


if __name__ == "__main__":
    main()
