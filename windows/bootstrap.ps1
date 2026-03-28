param(
    [string]$Distro = "Ubuntu",
    [string]$BridgeDir = "~/.gpu-bridge",
    [string]$LanSubnet = "192.168.0.0/16"
)

$ErrorActionPreference = "Stop"

function Write-Json($obj) {
    $obj | ConvertTo-Json -Depth 8 -Compress
}

$result = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    distro = $Distro
    bridge_dir = $BridgeDir
    steps = @()
    ok = $true
}

function Add-Step($name, $ok, $detail) {
    $step = [ordered]@{ name = $name; ok = $ok; detail = $detail }
    $result.steps += $step
    if (-not $ok) { $result.ok = $false }
}

try {
    $sshd = Get-Service -Name sshd -ErrorAction SilentlyContinue
    if (-not $sshd) {
        Add-Step "sshd-service" $false "OpenSSH Server (sshd) service not found"
    } else {
        Set-Service -Name sshd -StartupType Automatic
        if ($sshd.Status -ne "Running") { Start-Service sshd }
        Add-Step "sshd-service" $true "sshd running"
    }

    $sshRuleName = "OpenSSH-LAN-Only"
    Get-NetFirewallRule -DisplayName $sshRuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule | Out-Null
    New-NetFirewallRule `
        -DisplayName $sshRuleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 22 `
        -RemoteAddress $LanSubnet `
        -Profile Private `
        -Program System | Out-Null
    Get-NetFirewallRule -DisplayName "OpenSSH SSH Server (sshd)" -ErrorAction SilentlyContinue |
        Set-NetFirewallRule -Enabled False | Out-Null
    Add-Step "ssh-firewall" $true "sshd restricted to $LanSubnet on private profile"

    $wslCheck = & wsl.exe -l -q 2>$null
    if (-not $wslCheck) {
        Add-Step "wsl-distro-check" $false "No WSL distros detected"
    } else {
        Add-Step "wsl-distro-check" $true "Detected distros: $($wslCheck -join ', ')"
    }

    $health = & wsl.exe -d $Distro -- bash -lc "echo WSL_OK && python3 --version" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Add-Step "wsl-health" $false ($health | Out-String)
    } else {
        Add-Step "wsl-health" $true ($health | Out-String)
    }

    $bridgeDirBash = $BridgeDir
    if ($bridgeDirBash.StartsWith("~/")) {
        $bridgeDirBash = "/home/$env:USERNAME/" + $bridgeDirBash.Substring(2)
    } elseif ($bridgeDirBash -eq "~") {
        $bridgeDirBash = "/home/$env:USERNAME"
    }
    $bootScript = "if [ -x '$bridgeDirBash/boot.sh' ]; then '$bridgeDirBash/boot.sh' '$bridgeDirBash'; else echo bridge_boot_missing; fi"
    $bootOut = & wsl.exe -d $Distro -- bash -lc $bootScript 2>&1
    if ($LASTEXITCODE -ne 0) {
        Add-Step "bridge-boot" $false ($bootOut | Out-String)
    } else {
        Add-Step "bridge-boot" $true ($bootOut | Out-String)
    }

    $workerCheck = & wsl.exe -d $Distro -- pgrep -f "$bridgeDirBash/worker.py" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $workerArgs = "-d $Distro -- python3 $bridgeDirBash/worker.py --db $bridgeDirBash/state/bridge.db"
        Start-Process -WindowStyle Hidden -FilePath "wsl.exe" -ArgumentList $workerArgs | Out-Null
        Start-Sleep -Seconds 1
        $workerCheck2 = & wsl.exe -d $Distro -- pgrep -f "$bridgeDirBash/worker.py" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Add-Step "bridge-worker" $true "worker started via detached wsl.exe process"
        } else {
            Add-Step "bridge-worker" $false "worker process not detected after launch"
        }
    } else {
        Add-Step "bridge-worker" $true "worker already running"
    }

    # Power hardening for headless jobs on AC power.
    & powercfg /change standby-timeout-ac 0 | Out-Null
    & powercfg /change hibernate-timeout-ac 0 | Out-Null
    & powercfg /change monitor-timeout-ac 15 | Out-Null
    Add-Step "power-policy" $true "AC sleep/hibernate disabled, monitor timeout set"
}
catch {
    Add-Step "exception" $false $_.Exception.Message
}

Write-Output (Write-Json $result)
if (-not $result.ok) { exit 1 }
