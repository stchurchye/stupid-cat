# 注册 Windows 计划任务：开机登录自启 + 每日清理旧录像
# 用法（PowerShell）：.\scripts\install_scheduled_tasks.ps1
# 卸载：.\scripts\install_scheduled_tasks.ps1 -Uninstall

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StartScript = Join-Path $Root "scripts\start.ps1"
$PruneScript = Join-Path $Root "scripts\prune_recordings.ps1"
$TaskServe = "StupidCat-Serve"
$TaskPrune = "StupidCat-PruneRecordings"
$User = $env:USERNAME

function Remove-TaskIfExists([string]$Name) {
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "Removed task: $Name"
    }
}

if ($Uninstall) {
    Remove-TaskIfExists $TaskServe
    Remove-TaskIfExists $TaskPrune
    Write-Host "Scheduled tasks uninstalled."
    exit 0
}

foreach ($path in @($StartScript, $PruneScript)) {
    if (-not (Test-Path $path)) {
        throw "Missing script: $path"
    }
}

Remove-TaskIfExists $TaskServe
Remove-TaskIfExists $TaskPrune

$psExe = (Get-Command powershell.exe).Source
$serveArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`""
$pruneArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$PruneScript`" -Days 30"

$serveAction = New-ScheduledTaskAction `
    -Execute $psExe `
    -Argument $serveArgs `
    -WorkingDirectory $Root

$serveTrigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$serveTrigger.Delay = "PT1M"

$serveSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask `
    -TaskName $TaskServe `
    -Action $serveAction `
    -Trigger $serveTrigger `
    -Settings $serveSettings `
    -Description "Login autostart for stupid-cat (Web + dual RTSP)" `
    -RunLevel Limited | Out-Null

Write-Host "Registered: $TaskServe (At logon, 1 min delay)"

$pruneAction = New-ScheduledTaskAction `
    -Execute $psExe `
    -Argument $pruneArgs `
    -WorkingDirectory $Root

$pruneTrigger = New-ScheduledTaskTrigger -Daily -At "03:00"

$pruneSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskPrune `
    -Action $pruneAction `
    -Trigger $pruneTrigger `
    -Settings $pruneSettings `
    -Description "Delete stupid-cat recordings older than 30 days" `
    -RunLevel Limited | Out-Null

Write-Host "Registered: $TaskPrune (Daily 03:00, keep 30 days)"
Write-Host ""
Write-Host "Verify:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskServe','$TaskPrune' | Format-Table TaskName,State"
Write-Host "Uninstall:"
Write-Host "  .\scripts\install_scheduled_tasks.ps1 -Uninstall"
