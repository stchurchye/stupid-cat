# Operations: auto-start, retention & backup

## Auto-start on boot (with auto-restart)

The process handles SIGINT/SIGTERM gracefully (finalizes the active visit, releases
cameras), so it's safe to run under a service manager that restarts it on crash.

### Windows (Task Scheduler, runs at boot without login)

From an **elevated** PowerShell in the project directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_service.ps1
Start-ScheduledTask -TaskName stupid-cat       # start now without rebooting
```

It registers a task that runs `python -m stupid_cat serve` at startup as SYSTEM,
restarting every minute on failure. Remove with
`Unregister-ScheduledTask -TaskName stupid-cat -Confirm:$false`.
(For a true Windows *service* with the same effect, NSSM also works:
`nssm install stupid-cat <venv>\Scripts\python.exe -m stupid_cat serve`.)

### Linux (systemd)

Edit the paths in `deploy/stupid-cat.service`, then:

```bash
sudo cp deploy/stupid-cat.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now stupid-cat
```

---

# Retention & backup

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
python scripts/backup_db.py --zip-recordings --keep 14
```

### Windows Task Scheduler (daily 03:00)

```powershell
$py  = "C:\path\to\stupid-cat\.venv\Scripts\python.exe"
$cwd = "C:\path\to\stupid-cat"
$action  = New-ScheduledTaskAction -Execute $py -Argument "scripts\backup_db.py --keep 14" -WorkingDirectory $cwd
$trigger = New-ScheduledTaskTrigger -Daily -At 3am
Register-ScheduledTask -TaskName "stupid-cat-backup" -Action $action -Trigger $trigger -Description "Daily stupid-cat DB backup"
```

Point `--dest` at a different drive / network share (or sync `backups/` to cloud)
so a disk failure doesn't take the backups with it.

### cron (Linux, daily 03:00)

```cron
0 3 * * * cd /path/to/stupid-cat && .venv/bin/python scripts/backup_db.py --keep 14
```
