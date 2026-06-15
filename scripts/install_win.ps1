# Win1060 setup helper — cu121 torch + project deps
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$tmp = Join-Path $Root ".tmp"
$cache = Join-Path $Root ".pip-cache"
New-Item -ItemType Directory -Force -Path $tmp, $cache, (Join-Path $Root "models"), (Join-Path $Root "data") | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp
$env:PIP_CACHE_DIR = $cache
$env:HTTP_PROXY = ''
$env:HTTPS_PROXY = ''
$env:http_proxy = ''
$env:https_proxy = ''

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    py -3.12 -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-win-cuda.txt
pip install -e ".[dev]"

# YOLO weights (~18MB)
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
    Write-Host "Created config.local.yaml — edit RTSP URLs, device, and ROI."
}

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  1. Edit config.local.yaml (RTSP URLs, device cuda:0, ROI)"
Write-Host "  2. Optional: install ffmpeg and add to PATH (browser playback)"
Write-Host "  3. Start: .\scripts\start.ps1  or  python -m stupid_cat serve"
