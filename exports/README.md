# Data exports

Snapshots of `data/` for syncing between machines (not for secrets).

## Snapshot `2026-06-15`

- **7 visits** (2026-06-15 evening), all `cat_id=unknown` (no refs/centroids yet)
- Dual-camera **recordings** (`*_cam2.mp4` pairs)
- **correction_crops** from visits
- `stupid_cat.db` (visits + cat registry)

Does **not** include `config.local.yaml` (RTSP passwords stay local).

### Restore on another machine

```powershell
cd stupid-cat
git pull
# stop serve if running
New-Item -ItemType Directory -Force -Path data | Out-Null
Copy-Item -Recurse -Force exports\snapshots\2026-06-15\* data\
python -m stupid_cat serve --config config.yaml --local-config config.local.yaml
```

Browse timeline/stats at http://127.0.0.1:8765/ after restore.

### Local zip (optional, not in git — >100MB)

`exports/stupid-cat-data-2026-06-15.zip` can be recreated from `exports/snapshots/2026-06-15/`.
