param(
    [string]$Distro = "Ubuntu",
    [string]$LinuxUser = "",
    [switch]$RestartWsl = $true
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Write-Json($obj) {
    $obj | ConvertTo-Json -Depth 10 -Compress
}

$result = [ordered]@{
    ok = $true
    distro = $Distro
    linux_user = $LinuxUser
    steps = @()
}

function Add-Step($name, $ok, $detail) {
    $step = [ordered]@{ name = $name; ok = $ok; detail = $detail }
    $result.steps += $step
    if (-not $ok) { $result.ok = $false }
}

try {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $distroList = & wsl.exe -l -q 2>$null
    $distroRc = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if (-not $distroList) {
        throw "No WSL distributions found"
    }
    if ($distroRc -ne 0) {
        throw "wsl.exe list failed with exit code $distroRc"
    }

    if ([string]::IsNullOrWhiteSpace($LinuxUser)) {
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $detected = (& wsl.exe -d $Distro -- bash -lc "id -un" 2>$null)
        $detectedRc = $LASTEXITCODE
        $ErrorActionPreference = $oldPreference
        if (-not $detected) {
            throw "Could not detect Linux default user in distro '$Distro'"
        }
        if ($detectedRc -ne 0) {
            throw "Failed to detect Linux user in distro '$Distro' (exit code $detectedRc)"
        }
        $LinuxUser = $detected.Trim()
        $result.linux_user = $LinuxUser
    }

    $bashScript = @'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LINUX_USER="__LINUX_USER__"

apt-get update
apt-get install -y ca-certificates curl gnupg docker.io

install -m 0755 -d /etc/apt/keyrings /etc/apt/sources.list.d
if [ ! -f /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
fi
chmod a+r /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker || true

# Ensure systemd boot flag exists while preserving other sections/settings.
python3 - <<'PY'
from pathlib import Path

p = Path("/etc/wsl.conf")
if p.exists():
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
else:
    lines = []

out = []
current = None
has_boot = False
has_systemd = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        if current == "boot" and not has_systemd:
            out.append("systemd=true")
            has_systemd = True
        current = stripped[1:-1].strip().lower()
        if current == "boot":
            has_boot = True
        out.append(line)
        continue

    if current == "boot":
        normalized = stripped.replace(" ", "").lower()
        if normalized == "systemd=true":
            has_systemd = True
    out.append(line)

if has_boot and not has_systemd:
    out.append("systemd=true")

if not has_boot:
    if out and out[-1].strip() != "":
        out.append("")
    out.extend(["[boot]", "systemd=true"])

p.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY

if id -u "$LINUX_USER" >/dev/null 2>&1; then
  usermod -aG docker "$LINUX_USER" || true
  USER_HOME="$(getent passwd "$LINUX_USER" | cut -d: -f6)"
  if [ -n "${USER_HOME:-}" ]; then
    install -d -m 0700 -o "$LINUX_USER" -g "$LINUX_USER" "$USER_HOME/.docker"
    if [ -f "$USER_HOME/.docker/config.json" ]; then
      python3 - "$USER_HOME/.docker/config.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    data = {}
data.pop("credsStore", None)
data.pop("credStore", None)
p.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
PY
    else
      printf '{}\n' > "$USER_HOME/.docker/config.json"
    fi
    chown "$LINUX_USER:$LINUX_USER" "$USER_HOME/.docker/config.json"
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl enable docker || true
  systemctl restart docker || true
fi

echo "NATIVE_RUNTIME_READY"
'@

    $bashScript = $bashScript.Replace("__LINUX_USER__", $LinuxUser)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bashScript))
    $execCmd = "echo '$encoded' | base64 -d > /tmp/gpu-bridge-native-runtime.sh && bash /tmp/gpu-bridge-native-runtime.sh"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $runtimeOut = & wsl.exe -d $Distro -u root -- bash -lc $execCmd 2>&1
    $runtimeRc = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($runtimeRc -ne 0) {
        Add-Step "native-runtime-install" $false ($runtimeOut | Out-String)
    } else {
        Add-Step "native-runtime-install" $true ($runtimeOut | Out-String)
    }

    if ($RestartWsl) {
        & wsl.exe --shutdown | Out-Null
        Start-Sleep -Seconds 2
        Add-Step "wsl-restart" $true "wsl --shutdown completed"
    }

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $verify = & wsl.exe -d $Distro -- bash -lc "ps -p 1 -o comm=; docker --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1" 2>&1
    $verifyRc = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($verifyRc -ne 0) {
        Add-Step "verify-runtime" $false ($verify | Out-String)
    } else {
        Add-Step "verify-runtime" $true ($verify | Out-String)
    }
}
catch {
    Add-Step "exception" $false $_.Exception.Message
}

Write-Output (Write-Json $result)
if (-not $result.ok) { exit 1 }
