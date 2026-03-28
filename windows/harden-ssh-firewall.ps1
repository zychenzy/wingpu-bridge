param(
    [string]$LanSubnet = "192.168.0.0/16",
    [string]$RuleName = "OpenSSH-LAN-Only"
)

$ErrorActionPreference = "Stop"

# Ensure sshd exists and starts automatically.
$sshd = Get-Service -Name sshd -ErrorAction SilentlyContinue
if (-not $sshd) {
    throw "OpenSSH Server service (sshd) not found"
}
Set-Service -Name sshd -StartupType Automatic
if ($sshd.Status -ne "Running") { Start-Service sshd }

# Remove old rule if present.
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule | Out-Null

# Create a narrow inbound rule for SSH from LAN subnet only.
New-NetFirewallRule `
    -DisplayName $RuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 22 `
    -RemoteAddress $LanSubnet `
    -Profile Private `
    -Program System | Out-Null

# Optionally keep builtin rule disabled to avoid broad access.
Get-NetFirewallRule -DisplayName "OpenSSH SSH Server (sshd)" -ErrorAction SilentlyContinue |
  Set-NetFirewallRule -Enabled False | Out-Null

Write-Output (@{
    ok = $true
    lan_subnet = $LanSubnet
    rule = $RuleName
} | ConvertTo-Json -Compress)
