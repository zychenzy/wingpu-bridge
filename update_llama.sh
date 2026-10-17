#!/usr/bin/env bash
#
# Versioned llama.cpp upgrade for the wingpu runtime, with one-command rollback.
#
# Layout in WSL (paths under remote_src_root, default ~/src):
#   llama.cpp/                          main git checkout, used to fetch
#   llama.cpp/.worktrees/<tag>/         one git worktree per version (e.g. b11706),
#     build-cuda89/bin/llama-server     kept inside llama.cpp/ so the passwordless
#                                       /usr/local/sbin/wingpu-install-llamacpp-system
#                                       wrapper accepts it
#   llama-current  -> llama.cpp/.worktrees/<tag>/build-cuda89   (what wingpu runs)
#   llama-previous -> llama.cpp/.worktrees/<old>/build-cuda89   (rollback target)
#
# The runtime lanes in ~/.config/wingpu/wingpu.local.toml point server_bin at
# {remote_src_root}/llama-current/bin/llama-server, so an upgrade or rollback is
# just a symlink swap + `wingpu restart`. No config edit per version.
#
# Usage:
#   ./update_llama.sh                 build latest origin/master, smoke-test, switch, restart
#   ./update_llama.sh --ref b11706    build a specific tag/commit
#   ./update_llama.sh --no-switch     build + smoke-test only, leave the running version alone
#   ./update_llama.sh --rollback      swap llama-current and llama-previous, restart
#   ./update_llama.sh --jobs N        parallel build jobs (default: runtime_defaults.build_jobs)
#
# Old worktrees are never deleted automatically. Remove one with:
#   ssh HOST "wsl -d DISTRO -- git -C ~/src/llama.cpp worktree remove --force .worktrees/<tag>"

set -euo pipefail

REF="origin/master"; SWITCH=1; ROLLBACK=0; JOBS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --ref) REF="$2"; shift 2 ;;
    --no-switch) SWITCH=0; shift ;;
    --rollback) ROLLBACK=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "Usage: $0 [--ref REF] [--no-switch] [--rollback] [--jobs N]" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
command -v wingpu >/dev/null || { echo "wingpu CLI not in PATH" >&2; exit 1; }

# Use the wingpu tool's own interpreter (system python lacks its deps, e.g. gpubroker).
WPY=$(head -1 "$(command -v wingpu)" | sed 's/^#!//')
eval "$("$WPY" - <<'PY'
import sys, os, shlex
from wingpu_cli.main import load_settings
s = load_settings()
rd = s.runtime_defaults
print(f"HOST={shlex.quote(s.connection.host)}")
print(f"DISTRO={shlex.quote(s.connection.distro)}")
print(f"SRC_ROOT={shlex.quote(s.paths.remote_src_root)}")
print(f"CUDA_ARCH={shlex.quote(str(rd.cuda_architectures))}")
print(f"CMAKE_ARGS={shlex.quote(' '.join(rd.cmake_args))}")
print(f"DEF_JOBS={int(rd.build_jobs)}")
PY
)"
JOBS="${JOBS:-$DEF_JOBS}"

# Run a bash script inside WSL via stdin (avoids Windows cmd quoting).
wsl_run() { ssh "$HOST" "wsl -d $DISTRO -- bash -s" ; }

if [[ $ROLLBACK -eq 1 ]]; then
  echo "=== Rolling back: swapping llama-current <-> llama-previous ==="
  wsl_run <<EOF2
set -e
cd $SRC_ROOT
[[ -L llama-previous && -e llama-previous/bin/llama-server ]] || { echo "no usable llama-previous" >&2; exit 1; }
cur=\$(readlink llama-current); prev=\$(readlink llama-previous)
ln -sfn "\$prev" llama-current; ln -sfn "\$cur" llama-previous
echo "current  -> \$(readlink llama-current)"; echo "previous -> \$(readlink llama-previous)"
llama-current/bin/llama-server --version 2>&1 | head -1
sudo -n /usr/local/sbin/wingpu-install-llamacpp-system "\$(readlink -f llama-current)/bin" || echo "WARN: could not refresh /usr/local/bin copies"
EOF2
  wingpu restart && wingpu status
  exit 0
fi

echo "=== Building llama.cpp $REF on $HOST ($DISTRO) ==="
TAG=$(wsl_run <<EOF2
set -e
cd $SRC_ROOT/llama.cpp
git fetch --tags --quiet origin
git describe --tags --exact-match $REF 2>/dev/null || git describe --tags $REF 2>/dev/null | sed 's/-g.*//' || git rev-parse --short $REF
EOF2
)
TAG=$(echo "$TAG" | tail -1)
echo "Version tag: $TAG"

# Build in a transient systemd user unit so it survives the SSH/WSL session ending.
wsl_run <<EOF2
set -e
cd $SRC_ROOT
WT=$SRC_ROOT/llama.cpp/.worktrees/$TAG
grep -qx ".worktrees/" llama.cpp/.git/info/exclude || echo ".worktrees/" >> llama.cpp/.git/info/exclude
if [[ ! -d "\$WT" ]]; then git -C llama.cpp worktree add --detach "\$WT" $REF; fi
cd "\$WT"
systemctl --user reset-failed llama-build 2>/dev/null || true
systemd-run --user --wait --collect --unit=llama-build --working-directory="\$WT" bash -c \
  "cmake -S . -B build-cuda89 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH $CMAKE_ARGS > build.log 2>&1 && cmake --build build-cuda89 --config Release -j $JOBS --target llama-server llama-bench >> build.log 2>&1"
tail -3 build.log
build-cuda89/bin/llama-server --version 2>&1 | head -1
EOF2

if [[ $SWITCH -eq 0 ]]; then echo "Built $TAG; not switching (--no-switch)."; exit 0; fi

echo "=== Switching llama-current -> llama.cpp-$TAG (previous kept for rollback) ==="
wingpu stop
wsl_run <<EOF2
set -e
cd $SRC_ROOT
if [[ -L llama-current ]] && [[ "\$(readlink llama-current)" != "$SRC_ROOT/llama.cpp/.worktrees/$TAG/build-cuda89" ]]; then
  ln -sfn "\$(readlink llama-current)" llama-previous
fi
ln -sfn $SRC_ROOT/llama.cpp/.worktrees/$TAG/build-cuda89 llama-current
echo "current  -> \$(readlink llama-current)"; echo "previous -> \$(readlink llama-previous 2>/dev/null || echo none)"
sudo -n /usr/local/sbin/wingpu-install-llamacpp-system "\$(readlink -f llama-current)/bin" || echo "WARN: could not refresh /usr/local/bin copies"
EOF2
wingpu start
wingpu status
echo
echo "Done. If anything looks wrong: $0 --rollback"
