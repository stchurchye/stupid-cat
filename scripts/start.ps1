# 启动 stupid-cat 服务（Web + 双路 RTSP）
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $Root

$env:TMP = Join-Path $Root ".tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

$ffmpegBin = Join-Path $Root "tools\ffmpeg\bin"
if (Test-Path (Join-Path $ffmpegBin "ffmpeg.exe")) {
    $env:PATH = "$ffmpegBin;$env:PATH"
}

$portLine = netstat -ano | Select-String ":8765" | Select-String "LISTENING" | Select-Object -First 1
if ($portLine) {
    $procId = ($portLine -split "\s+")[-1]
    if ($procId -match '^\d+$') {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
        $cmd = $proc.CommandLine
        if ($cmd -and $cmd -like '*stupid_cat*') {
            Write-Host "Stopping existing stupid-cat on port 8765 (PID $procId)..."
            Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        } else {
            Write-Host "Port 8765 in use by another program (PID $procId); not stopping it."
        }
    }
}

.\.venv\Scripts\Activate.ps1
python -m stupid_cat serve --config config.yaml --local-config config.local.yaml
