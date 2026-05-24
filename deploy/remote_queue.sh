#!/usr/bin/env bash
# Run a LIST of experiments SEQUENTIALLY, detached (survives SSH logout).
# One GPU -> run jobs back-to-back rather than thrashing in parallel.
# Master log: logs/queue_<timestamp>.log ; per-run: logs/<exp>_s<seed>.log
#
# Usage (via deploy/remote.ps1 -Action queue, or directly):
#   REMOTE_VENV=hw3 bash deploy/remote_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

# --- EDIT THIS QUEUE -------------------------------------------------------
# Each entry is "<config_basename> <seed>". Tier A x3 seeds shown by default;
# uncomment Tier B lines to queue the full scope.
LIST=(
  # Round 2: V0 (entropy fix) + P1-P3 reruns at 7M + Tier B.
  # V1/V2 already succeeded at 2M and are NOT re-run.
  # Order: quick checks first, then cheap PPO Tier-B, then the long 7M DQN grind,
  # so the fast/insurance results are banked early on the shared GPU.
  "V0_basic 0" "V0_basic 1" "V0_basic 2"
  "P1 0"
  "P5_ppo_pong 0" "P5_ppo_pong 1" "P5_ppo_pong 2"
  "V3_healthgathering 0" "V3_healthgathering 1" "V3_healthgathering 2"
  "V4_stack1 0" "V4_stack1 1" "V4_stack1 2"
  "P1 1" "P1 2"
  "P2_targetoff 0" "P2_targetoff 1" "P2_targetoff 2"
  "P3_epsfast 0" "P3_epsfast 1" "P3_epsfast 2"
  "P3_epsslow 0" "P3_epsslow 1" "P3_epsslow 2"
  "P4_buffersmall 0" "P4_buffersmall 1" "P4_buffersmall 2"
  "V5_dqn_defendcenter 0" "V5_dqn_defendcenter 1" "V5_dqn_defendcenter 2"
)
# ---------------------------------------------------------------------------

map_script() {
  local cfg="$1" task algo
  task="$(awk '/^task:/{print $2; exit}' "$cfg")"
  algo="$(awk '/^algo:/{print $2; exit}' "$cfg")"
  case "${task}_${algo}" in
    pong_dqn)    echo "pong/train_pong_dqn.py" ;;
    pong_ppo)    echo "pong/train_pong_ppo.py" ;;
    vizdoom_ppo) echo "vizdoom/train_vizdoom_ppo.py" ;;
    vizdoom_dqn) echo "vizdoom/train_vizdoom_dqn.py" ;;
    *) return 1 ;;
  esac
}

run_queue() {
  source deploy/_activate.sh
  activate_env
  local n=${#LIST[@]} i=0
  for entry in "${LIST[@]}"; do
    i=$((i + 1))
    # shellcheck disable=SC2086
    set -- $entry
    local name="$1" seed="$2" cfg="configs/$1.yaml" script rlog
    if [ ! -f "$cfg" ]; then echo "[$i/$n] SKIP $name: $cfg not found"; continue; fi
    if ! script="$(map_script "$cfg")"; then echo "[$i/$n] SKIP $name: cannot map task/algo"; continue; fi
    rlog="logs/${name}_s${seed}.log"
    echo "=== [$i/$n] $(date '+%F %T') START $name seed=$seed -> $rlog ==="
    if python -u "$script" --config "$cfg" --seed "$seed" > "$rlog" 2>&1; then
      echo "=== [$i/$n] $(date '+%F %T') DONE  $name seed=$seed ==="
    else
      echo "=== [$i/$n] $(date '+%F %T') FAIL  $name seed=$seed (see $rlog) ==="
    fi
  done
  echo "QUEUE_DONE"
}

# Re-exec a detached worker so the whole queue survives SSH logout.
if [ "${1:-}" = "--worker" ]; then run_queue; exit 0; fi

ts="$(date +%Y%m%d_%H%M%S)"
qlog="logs/queue_${ts}.log"
nohup bash "$0" --worker > "$qlog" 2>&1 < /dev/null &
echo $! > logs/queue.pid
echo "STARTED queue (${#LIST[@]} runs)  pid=$(cat logs/queue.pid)"
echo "  master log: $qlog    (track:  tail -f $qlog)"
echo "  per-run logs: logs/<exp>_s<seed>.log"
