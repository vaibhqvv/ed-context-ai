#!/usr/bin/env bash
# Container-only GPU cloud bootstrap for ED-Context-AI.
# Run this once after you shell into the container.
#
# Required artifacts (bring these into the container):
#   - source tree (clone or rsync this repo)
#   - outputs/models/best_model.pt   (model checkpoint, ~717 KB)
#   - config.yaml                    (already in repo root)
#
# Exposed port:
#   $PORT (default 8000). Point your cloud's HTTP tunnel at this port.

set -euo pipefail

# --- Resolve project root (this script lives in deploy/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# --- Python environment ---
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- GPU / runtime flags ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Keep 1 worker so the GRU is loaded once on the GPU
export WORKERS="${WORKERS:-1}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

# --- Optional API key (set API_KEY env var to enable bearer auth) ---
# export API_KEY="change-me"

echo "[setup] installing serving dependencies..."
$PYTHON_BIN -m pip install --no-cache-dir --upgrade pip
$PYTHON_BIN -m pip install --no-cache-dir -r deploy/requirements-serve.txt

echo "[setup] verifying checkpoint..."
if [[ ! -f "outputs/models/best_model.pt" ]]; then
    echo "[error] outputs/models/best_model.pt not found." >&2
    echo "        Upload/mount your trained checkpoint before starting the server." >&2
    exit 1
fi

echo "[setup] verifying GPU..."
$PYTHON_BIN - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
PY

echo "[run] starting ED-Context-AI on ${HOST}:${PORT} (workers=${WORKERS})"
exec $PYTHON_BIN -m uvicorn deploy.serve:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info
