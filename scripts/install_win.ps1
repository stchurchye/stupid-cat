# Win1060 setup helper (Task 15)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win-cuda.txt
pip install -e .

Write-Host "Copy config.local.yaml from config.yaml and set RTSP URLs + device cuda:0"
Write-Host "Download yolo11s.pt to models/ and run: python -m stupid_cat --config config.yaml"
