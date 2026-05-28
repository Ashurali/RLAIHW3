#!/usr/bin/env bash
# Round 3 catch-up queue: runs AFTER V5 finishes.
#   - P4_buffersmall × 3 seeds at 2M (T2 small-buffer gap)
#   - P5b_ppo_zoo × 3 seeds at 7M (T1 PPO literature-recipe sanity check)
#
# Same nohup / detached pattern as remote_queue.sh. Launched by the V5 watcher
# when the round-2 queue master exits.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

LIST=(
  # Cheap small-buffer ablation first (2M, single-env DQN, ~30-60min/seed).
  "P4_buffersmall 0" "P4_buffersmall 1" "P4_buffersmall 2"
  # Literature-recipe PPO last (7M, 16 envs, ~2h/seed at idle GPU).
  "P5b_ppo_zoo 0" "P5b_ppo_zoo 1" "P5b_ppo_zoo 2"
)

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

if [ "${1:-}" = "--worker" ]; then run_queue; exit 0; fi

ts="$(date +%Y%m%d_%H%M%S)"
qlog="logs/queue_round3_${ts}.log"
nohup bash "$0" --worker > "$qlog" 2>&1 < /dev/null &
echo $! > logs/queue_round3.pid
echo "STARTED round3 queue (${#LIST[@]} runs)  pid=$(cat logs/queue_round3.pid)"
echo "  master log: $qlog    (track:  tail -f $qlog)"
echo "  per-run logs: logs/<exp>_s<seed>.log"
