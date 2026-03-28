#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-gpu-host}"
DISTRO="${2:-Ubuntu}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
OUT_DIR="${3:-${REPO_ROOT}/bridge/remote-only/wsl-llama}"

REMOTE_HOME="$(ssh "${HOST}" "wsl -d ${DISTRO} -- bash -lc 'printf %s \"\$HOME\"'")"
REMOTE_SRC_ROOT="${REMOTE_HOME}/src"
REMOTE_MODELS_ROOT="${REMOTE_HOME}/models/Qwen"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/metadata" "${OUT_DIR}/src" "${OUT_DIR}/home/remote-user" "${OUT_DIR}/system"

ssh_wsl() {
  ssh "${HOST}" "wsl -d ${DISTRO} -- env REMOTE_HOME='${REMOTE_HOME}' REMOTE_SRC_ROOT='${REMOTE_SRC_ROOT}' REMOTE_MODELS_ROOT='${REMOTE_MODELS_ROOT}' bash -s"
}

cat > "${OUT_DIR}/README.md" <<EOF2
# Remote WSL Llama Mirror

This directory is a repo-local mirror of the remote WSL llama setup on:

- host: \`${HOST}\`
- distro: \`${DISTRO}\`

Included:

- source trees for \`llama.cpp\` and \`llama-cpp-turboquant-cuda\`
- \`~/.gpu-bridge\` runtime state and logs
- narrow admin wrapper scripts from \`/usr/local/sbin\`
- metadata snapshots for git state, runtime state, model inventory, and system info

Excluded:

- model weights under \`${REMOTE_MODELS_ROOT}\`
- build directories such as \`build-cuda89\`
- git object directories such as \`.git\`

Regenerate with:

\`\`\`bash
./bridge/mac/mirror_remote_wsl_llama.sh ${HOST} ${DISTRO}
\`\`\`
EOF2

ssh_wsl <<'EOF' > "${OUT_DIR}/metadata/system.txt"
set -euo pipefail
echo "snapshot_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname)"
echo "user=$(whoami)"
echo "pwd=$(pwd)"
echo "uname=$(uname -a)"
echo
echo "[nvidia-smi]"
nvidia-smi || true
echo
echo "[cuda]"
command -v nvcc >/dev/null 2>&1 && nvcc --version || true
EOF

ssh_wsl <<'EOF' > "${OUT_DIR}/metadata/git_state.txt"
set -euo pipefail
for repo in ${REMOTE_SRC_ROOT}/llama.cpp ${REMOTE_SRC_ROOT}/llama-cpp-turboquant-cuda; do
  if [[ -d "$repo/.git" ]]; then
    echo "== $repo =="
    git -C "$repo" remote -v | sed -n '1,4p'
    echo "branch=$(git -C "$repo" branch --show-current)"
    echo "commit=$(git -C "$repo" rev-parse HEAD)"
    echo "status:"
    git -C "$repo" status --short || true
    echo
  fi
done
EOF

ssh_wsl <<'EOF' > "${OUT_DIR}/metadata/runtime_state.txt"
set -euo pipefail
echo "[gpu-bridge tree]"
find ~/.gpu-bridge -maxdepth 2 -type f | sort || true
echo
echo "[gpu-bridge selected files]"
for file in ~/.gpu-bridge/selected_*; do
  [[ -f "$file" ]] || continue
  echo "== $file =="
  cat "$file"
done
echo
echo "[active llama processes]"
ps -ef | grep llama | grep -v grep || true
echo
echo "[ports]"
ss -ltnp | grep ':8000' || true
EOF

ssh_wsl <<'EOF' > "${OUT_DIR}/metadata/model_inventory.txt"
set -euo pipefail
echo "[directories]"
find ${REMOTE_MODELS_ROOT} -maxdepth 2 -type d | sort || true
echo
echo "[gguf files]"
find ${REMOTE_MODELS_ROOT} -maxdepth 2 -type f -name '*.gguf' | sort || true
EOF

ssh_wsl <<'EOF' > "${OUT_DIR}/metadata/help_turbo.txt"
set -euo pipefail
${REMOTE_SRC_ROOT}/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-server --help 2>&1 | grep -i -E 'cache-type-k|cache-type-v|turbo|flash-attn' || true
EOF

ssh_wsl <<'EOF' | tar -xf - -C "${OUT_DIR}/src"
set -euo pipefail
cd ${REMOTE_SRC_ROOT}
tar \
  --exclude='.git' \
  --exclude='build' \
  --exclude='build-*' \
  --exclude='build_*' \
  -cf - \
  llama.cpp \
  llama-cpp-turboquant-cuda
EOF

ssh_wsl <<'EOF' | tar -xf - -C "${OUT_DIR}/home/remote-user"
set -euo pipefail
cd ${REMOTE_HOME}
tar -cf - .gpu-bridge
EOF

ssh_wsl <<'EOF' | tar -xf - -C "${OUT_DIR}/system"
set -euo pipefail
cd /
tar -cf - \
  usr/local/sbin/wingpu-install-build-prereqs \
  usr/local/sbin/wingpu-install-cuda-toolkit \
  usr/local/sbin/wingpu-install-experiment-prereqs \
  usr/local/sbin/wingpu-install-llamacpp-system
EOF

echo "Mirrored remote WSL llama setup to: ${OUT_DIR}"
