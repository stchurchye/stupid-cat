<#
.SYNOPSIS
  Install stupid-cat as a boot-start Windows scheduled task with auto-restart.
.DESCRIPTION
  Registers a task that runs `python -m stupid_cat serve` at system startup (no
  login required), restarting on crash. Run from an elevated PowerShell in the
  project directory:  powershell -ExecutionPolicy Bypass -File scripts\install_service.ps1
  Remove with:  Unregister-ScheduledTask -TaskName stupid-cat -Confirm:$false
#>
param(
  [string]$ProjectDir = (Get-Location).Path,
  [string]$Python = "",
  [string]$TaskName = "stupid-cat"
)

if (-not $Python) { $Python = Join-Path $ProjectDir ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) { throw "python not found at $Python (create the venv first, or pass -Python)" }

$args = "-m stupid_cat serve --config config.yaml --local-config config.local.yaml"
$action  = New-ScheduledTaskAction -Execute $Python -Argument $args -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger -AtStartup
# Auto-restart on crash; no run-time limit; start even if the trigger was missed.
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
# Run as SYSTEM so it starts without anyone logging in.
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
  -Principal $principal -Force -Description "stupid-cat litter vision monitor (auto-start at boot)"

Write-Host "Installed scheduled task '$TaskName'. Start now with: Start-ScheduledTask -TaskName $TaskName"
