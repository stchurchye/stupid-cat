# 删除超过 N 天的 visit 录像（默认 30 天），并清理数据库中的 recording_path
param(
    [int]$Days = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$LogDir = Join-Path $Root ".tmp"
$LogFile = Join-Path $LogDir "prune_recordings.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

.\.venv\Scripts\python.exe scripts\prune_recordings.py --days $Days *>&1 |
    Tee-Object -FilePath $LogFile -Append
