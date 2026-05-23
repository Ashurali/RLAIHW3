#!/usr/bin/env bash
# One-time server setup, DETACHED (survives SSH logout). Probes the hardware,
# creates the venv, installs requirements, and verifies CUDA.
#
# Usage (via deploy/remote.ps1 -Action setup, or directly on the server):
#   REMOTE_VENV=hw3 bash deploy/remote_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

run_setup() {
  echo "=== server probe ($(date '+%F %T')) ==="
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not found"; fi
  echo "CPUs(threads): $(nproc 2>/dev/null || echo '?')"
  echo "RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
  echo "Python(default): $(python3 --version 2>&1)"
  echo "Available interpreters:"
  for p in python3.12 python3.11 python3.10; do
    command -v "$p" >/dev/null 2>&1 && echo "  - $p = $($p --version 2>&1)"
  done

  # Prefer 3.12/3.11 over the conda base (3.13): broadest torch/vizdoom wheel
  # coverage. Override with REMOTE_PY_BIN in deploy/server.env if needed.
  if [ -z "${PY_BIN:-}" ]; then
    for p in python3.12 python3.11 python3.10 python3; do
      if command -v "$p" >/dev/null 2>&1; then PY_BIN="$p"; break; fi
    done
  fi
  echo "=== creating venv: $VENV  (interpreter: $PY_BIN = $($PY_BIN --version 2>&1)) ==="
  "$PY_BIN" -m venv "$VENV" || { echo "venv failed -- you may need: sudo apt install ${PY_BIN}-venv" >&2; exit 1; }
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip wheel

  echo "=== installing requirements (CUDA torch via extra-index-url) ==="
  pip install -r requirements.txt

  echo "=== verifying torch sees the GPU ==="
  python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
    print("cuda runtime:", torch.version.cuda)
else:
    print("WARNING: CUDA not available -- training will fall back to CPU.")
PY
  echo "SETUP_DONE"
}

# Re-exec a detached worker so the install survives SSH logout.
if [ "${1:-}" = "--worker" ]; then run_setup; exit 0; fi

log="logs/setup.log"
nohup bash "$0" --worker > "$log" 2>&1 < /dev/null &
echo $! > logs/setup.pid
echo "STARTED setup  pid=$(cat logs/setup.pid)"
echo "  log: $log    (wait for SETUP_DONE:  tail -f $log)"
