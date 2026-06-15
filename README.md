# stupid-cat

Computer-vision monitor for five cats sharing a large lidded litter box: **who**
used it, **when**, for **how long**, **pee vs poop**, and the **video** — across
two cameras, day (colour) and night (IR).

**Spec:** `docs/superpowers/specs/2026-06-02-stupid-cat-litter-vision-design.md`

## How it works

```
RTSP cam1 ┐
          ├─▶ motion-gated ingest ─▶ YOLO detect ─▶ ROI overlap ─▶ Visit FSM ─┐
RTSP cam2 ┘                                                                   │
                                          ┌──────────────────────────────────┘
                                          ▼
   per-visit: record clips ┄ buffer Re-ID embeddings (both cams, fused once) ┄ digging signal
                                          ▼
                       identity (gallery + daytime colour) · pee/poop · multi-cat
                                          ▼
                            SQLite  ──▶  FastAPI + web UI (review / correct)
```

- **Detection** — YOLO11s. A hunched/digging cat is often misread as sheep/dog/etc.,
  and a litter box only holds a cat, so a configurable set of animal classes counts
  (`inference.detect_class_ids`) to avoid splitting one visit into several.
- **Visit FSM** — `idle → active → cooldown`, gated on ROI overlap. Tolerates a cat
  turning/digging without ending the visit; discards trips shorter than
  `min_visit_sec` (measured by real presence, not the exit-timeout window).
- **Identity (Re-ID)** — day/night-robust: embeddings are computed on **grayscale**
  so colour (day) and IR (night) share one feature space; each cat keeps a
  **multi-vector gallery** matched by top-k cosine; a **daytime colour histogram**
  only re-ranks cats that already clear the embedding threshold. Backbone is
  `efficientnet_b0` (default) or `dinov2_vits14/vitb14`. **Needs reference photos
  per cat** (see below) before it can name anyone.
- **Multi-cat** — if ≥2 cats are in the box for a sustained streak, identity is
  recorded as `unknown` (don't guess a blend) and the visit is flagged.
- **Pee/poop** — heuristic from duration + late-phase digging motion; tunable from
  real data (the digging signal is stored and `/api/v1/waste/accuracy` reports
  predicted-vs-corrected).

## Quick start (Mac, development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # GPU/Re-ID tests skip without torch
python -m stupid_cat run --video fixtures/sample_ir.mp4   # offline pipeline on a clip
python -m stupid_cat serve                  # pipeline + web UI at http://127.0.0.1:8765/
```

## Windows (production, GTX 1060 / Pascal)

```powershell
scripts\install_win.ps1            # venv + CUDA (cu121) torch + deps (Pascal-compatible pins)
Copy-Item config.yaml config.local.yaml    # then set RTSP URLs / device: cuda:0 in the local file
scripts\start.ps1                  # run once in the foreground to verify
```

CLI: `python -m stupid_cat run` (pipeline only) or `serve` (pipeline + API). Use
`--config` / `--local-config` to point at config files.

### Auto-start on boot

Two options (pick one):

- `scripts\install_scheduled_tasks.ps1` — registers **StupidCat-Serve** (starts at
  logon) + **StupidCat-PruneRecordings** (daily 03:00). Uninstall with `-Uninstall`.
- `scripts\install_service.ps1` — starts at **boot as SYSTEM** (no login needed)
  with **auto-restart on crash**. More robust for a headless box.

Linux: `deploy/stupid-cat.service` (systemd). See `docs/ops.md` for details.

## Teaching it your cats (Re-ID)

Identity is structural until you provide reference crops:

1. Collect clear crops of each cat — **both daytime colour and night IR** — and put
   them in `data/cats/<cat_id>/refs/` (the cat ids come from `cats.seed` in config).
2. Build galleries: `python scripts/build_embeddings.py` (mirrors the runtime
   grayscale + colour settings; writes `gallery.npy` / `color_gallery.npy` /
   `centroid.npy` per cat).
3. From then on, correcting a visit in the web UI moves its crop into that cat's
   refs and rebuilds the gallery automatically — accuracy improves as you correct.

Switching `reid_backbone` changes the embedding dimension, so **rebuild refs** after
a switch (stale-dimension galleries are skipped at match time).

## Configuration (`config.yaml`, override in `config.local.yaml`)

| Section | Key knobs |
|---|---|
| `service` | `host` (use `0.0.0.0` for LAN — then set `api_key`), `port`, `api_key`, `allowed_origins`, `trusted_hosts` |
| `inference` | `device`, `reid_backbone`, `reid_grayscale`, `reid_topk`, `color_weight`, `similarity_threshold`, `detect_class_ids`, `fp16` |
| `cameras` | per-camera `rtsp_url`, `weight`, normalized `roi_polygon` (0–1, resolution-independent) |
| `session` | `enter_overlap_sec`, `exit_no_cat_sec`, `cooldown_sec`, `min_visit_sec`, `roi_overlap_min` |
| `recorder` | `record_cameras`, `max_seconds`, `min_free_mb`, `retention_days` |
| `waste` | `enabled`, duration + `dig_motion_threshold` thresholds |
| `mqtt` | `enabled` (needs `pip install paho-mqtt`), broker, `topic_prefix` |

## Web UI

- `/` — visit timeline: time, cat (with 👥 multi-cat / 💩💧 waste badges), duration,
  clips, and dropdowns to correct cat id and pee/poop.
- `/stats.html` — per-cat and per-day summaries.
- `/live.html` — live camera previews with the ROI box.

## Operations

- **Recording retention** runs in-process (delete older than `retention_days`, and
  rotate the oldest out below `min_free_mb`). `scripts/prune_recordings.py` is the
  manual tool.
- **Backup**: `python scripts/backup_db.py --zip-recordings` (WAL-safe online copy).
- **Alerts** (optional MQTT): `visit_ended`, `alert/low_disk`, `alert/frame_errors`.

See `docs/ops.md` for scheduled-task / systemd install of auto-start and backup.

## Testing & CI

```bash
pytest -q
ruff check src tests scripts
mypy --ignore-missing-imports src/stupid_cat
```

GitHub Actions (`.github/workflows/ci.yml`) gates on **ruff + pytest** (GPU tests
skipped) **+ mypy**.

## Known limitations

- Re-ID needs the reference photos above; until then most visits are `unknown`.
- Cross-camera multi-cat (each cat only in one view) is detected only when
  `session.cameras_overlap: false` (disjoint cameras); the default assumes both
  cameras watch the same box.
- Pee/poop and identity thresholds are heuristics — tune against real data.
- `dinov2_*` downloads weights on first use (needs internet once).
