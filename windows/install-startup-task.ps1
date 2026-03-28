param(
    [string]$ScriptPath = "C:\ProgramData\gpu-bridge\bootstrap.ps1",
    [string]$TaskName = "GPUBridgeBoot",
    [string]$Distro = "Ubuntu",
    [string]$BridgeDir = "~/.gpu-bridge",
    [string]$RunAsUser = ""
)

$ErrorActionPreference = "Stop"

$taskDir = Split-Path -Parent $ScriptPath
if (-not (Test-Path $taskDir)) {
    New-Item -ItemType Directory -Path $taskDir -Force | Out-Null
}

$source = Join-Path $PSScriptRoot "bootstrap.ps1"
Copy-Item -Path $source -Destination $ScriptPath -Force

$effectiveUser = $RunAsUser
if ([string]::IsNullOrWhiteSpace($effectiveUser)) {
    $effectiveUser = "$env:COMPUTERNAME\$env:USERNAME"
}

$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -Distro `"$Distro`" -BridgeDir `"$BridgeDir`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId $effectiveUser -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Output (@{
    ok = $true
    task_name = $TaskName
    script_path = $ScriptPath
    distro = $Distro
    bridge_dir = $BridgeDir
    run_as_user = $effectiveUser
} | ConvertTo-Json -Compress)
