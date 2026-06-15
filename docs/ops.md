# Operations: retention & backup

## Recording retention (automatic, in-process)

The pipeline runs a rotation pass every ~10 minutes (no scheduler needed):

- `recorder.retention_days` — delete clips older than N days (`0` = keep forever).
- `recorder.min_free_mb` — when free disk falls below this, the oldest clips are
  deleted until it's satisfied (and new recording is paused until then).

Set both in `config.yaml` (or `config.local.yaml`). The standalone
`scripts/prune_recordings.py` remains for one-off manual cleanup.

## Database + recordings backup (scheduled)

`scripts/backup_db.py` makes a WAL-safe online copy of `data/stupid_cat.db`
(consistent even while the pipeline is writing) into `backups/<timestamp>/`, and
can also archive recordings. It prunes its own old backup folders (`--keep`).

```bash
python -m scripts.backup_db --zip-recordings --keep 14
```

### Windows Task Scheduler (daily 03:00)

```powershell
$py  = "C:\path\to\stupid-cat\.venv\Scripts\python.exe"
$cwd = "C:\path\to\stupid-cat"
$action  = New-ScheduledTaskAction -Execute $py -Argument "-m scripts.backup_db --keep 14" -WorkingDirectory $cwd
$trigger = New-ScheduledTaskTrigger -Daily -At 3am
Register-ScheduledTask -TaskName "stupid-cat-backup" -Action $action -Trigger $trigger -Description "Daily stupid-cat DB backup"
```

Point `--dest` at a different drive / network share (or sync `backups/` to cloud)
so a disk failure doesn't take the backups with it.

### cron (Linux, daily 03:00)

```cron
0 3 * * * cd /path/to/stupid-cat && .venv/bin/python -m scripts.backup_db --keep 14
```
