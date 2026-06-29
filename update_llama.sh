#!/usr/bin/env bash
#
# Script to update remote official llama.cpp to the latest from git repo,
# rebuild, install, and restart the wingpu runtime.
#

set -euo pipefail

# Parse command line arguments
CLEAN_ARG=""
JOBS_ARG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN_ARG="--clean"
      shift
      ;;
    --jobs)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --jobs requires an argument" >&2
        exit 1
      fi
      JOBS_ARG="--jobs $2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--clean] [--jobs <num_jobs>]" >&2
      exit 1
      ;;
  esac
done

echo "=== Extracting wingpu settings ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Retrieve settings using python from the wingpu package resources
SETTINGS_JSON=$(python3 -c "
import sys, os, json
sys.path.insert(0, os.path.abspath('mac/src'))
from wingpu_cli.main import load_settings, runtime_lane
settings = load_settings()
lane = runtime_lane(settings, 'upstream')
print(json.dumps({
    'host': settings.connection.host,
    'distro': settings.connection.distro,
    'source_dir': lane.source_dir
}))
")

HOST=$(echo "$SETTINGS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['host'])")
DISTRO=$(echo "$SETTINGS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['distro'])")
SOURCE_DIR=$(echo "$SETTINGS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['source_dir'])")

echo "Remote Host:   $HOST"
echo "WSL Distro:    $DISTRO"
echo "Source Dir:    $SOURCE_DIR"
echo ""

# Ensure wingpu is available
if ! command -v wingpu >/dev/null 2>&1; then
  echo "Error: wingpu CLI is not installed or not in PATH." >&2
  exit 1
fi

echo "=== Step 1: Stopping remote llama.cpp runtime ==="
wingpu stop

echo "=== Step 2: Fetching and pulling latest llama.cpp on remote WSL ==="
ssh "$HOST" "wsl -d $DISTRO -- git -C $SOURCE_DIR pull"

echo "=== Step 3: Rebuilding llama.cpp runtime ==="
wingpu build upstream $CLEAN_ARG $JOBS_ARG

echo "=== Step 4: Installing updated system binaries in WSL ==="
wingpu admin install-llamacpp-system upstream

echo "=== Step 5: Restarting the wingpu runtime ==="
wingpu start

echo "=== Step 6: Verifying status ==="
wingpu status

echo ""
echo "Update completed successfully!"
