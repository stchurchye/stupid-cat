# Win1060 setup helper
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$tmp = Join-Path $Root ".tmp"
$cache = Join-Path $Root ".pip-cache"
New-Item -ItemType Directory -Force -Path $tmp, $cache, (Join-Path $Root "models"), (Join-Path $Root "data") | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp
$env:PIP_CACHE_DIR = $cache

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    py -3.12 -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"

# YOLO 权重
if (-not (Test-Path "models\yolo11s.pt")) {
    python -c @"
from pathlib import Path
import shutil
from ultralytics.utils.downloads import attempt_download_asset
p = Path('models'); p.mkdir(exist_ok=True)
src = Path(attempt_download_asset('yolo11s.pt'))
shutil.copy2(src, p / 'yolo11s.pt')
print('Downloaded models/yolo11s.pt')
"@
}

if (-not (Test-Path "config.local.yaml")) {
    Copy-Item "config.local.yaml.example" "config.local.yaml"
    Write-Host "Created config.local.yaml — edit RTSP URLs and device."
}

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Edit config.local.yaml (RTSP URLs, device)"
Write-Host "  2. Optional GPU: .\scripts\install-cuda.ps1"
Write-Host "  3. Start: .\scripts\start.ps1  or  .\scripts\start-api-only.ps1"
