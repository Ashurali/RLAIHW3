#!/usr/bin/env bash
# Sourced by the remote deploy scripts to create/activate the training env.
# Supports two backends so it works with or without sudo:
#   ENV_KIND=venv   -> python -m venv  (needs python3.x-venv / ensurepip)
#   ENV_KIND=conda  -> conda env       (no system packages / sudo needed)
#
# Reads (with defaults) from the environment, passed by deploy/remote.ps1:
#   ENV_KIND          conda | venv                 (default: conda)
#   VENV              conda env name OR venv dir    (default: hw3 / REMOTE_VENV)
#   PY_VERSION        python for the conda env      (default: 3.12)
#   PY_BIN            interpreter for venv creation (auto-detect if empty)
#   REMOTE_CONDA_BASE conda base dir override       (auto-detect if empty)

VENV="${VENV:-${REMOTE_VENV:-hw3}}"
ENV_KIND="${ENV_KIND:-conda}"
PY_VERSION="${PY_VERSION:-3.12}"
PY_BIN="${PY_BIN:-}"

_ensure_conda() {
  command -v conda >/dev/null 2>&1 && return 0
  local c
  for c in "${REMOTE_CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" \
           "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda /opt/anaconda3; do
    if [ -n "$c" ] && [ -f "$c/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$c/etc/profile.d/conda.sh"
      return 0
    fi
  done
  echo "ERROR: cannot locate conda. Set REMOTE_CONDA_BASE in deploy/server.env" >&2
  echo "       to the output of 'conda info --base' on the server." >&2
  return 1
}

_pick_py() {
  local p
  for p in python3.12 python3.11 python3.10 python3; do
    command -v "$p" >/dev/null 2>&1 && { echo "$p"; return 0; }
  done
  return 1
}

# Activate an already-created environment into the current shell.
activate_env() {
  if [ "$ENV_KIND" = "conda" ]; then
    _ensure_conda || return 1
    conda activate "$VENV"
  else
    # shellcheck disable=SC1091
    conda activate "$VENV"
  fi
}

# Create the environment, then activate it.
create_env() {
  if [ "$ENV_KIND" = "conda" ]; then
    _ensure_conda || return 1
    echo "conda env: $VENV (python=$PY_VERSION) under $(conda info --base)"
    conda create -y -n "$VENV" "python=$PY_VERSION"
    conda activate "$VENV"
  else
    local py="${PY_BIN:-$(_pick_py)}"
    echo "venv interpreter: $py = $($py --version 2>&1)"
    "$py" -m venv "$VENV"
    # shellcheck disable=SC1091
    conda activate $VENV
  fi
}