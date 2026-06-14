# 仅 Web/API，不拉 RTSP（调试用）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$env:TMP = Join-Path $Root ".tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

.\.venv\Scripts\Activate.ps1
python -m stupid_cat serve --config config.yaml --local-config config.local.yaml --api-only
