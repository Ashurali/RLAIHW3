#!/usr/bin/env bash
# Launch ONE training run DETACHED on the server (survives SSH logout).
# Output goes to logs/<exp>_s<seed>.log so progress is trackable with tail -f.
#
# Usage (run from the repo root on the server, or via deploy/remote.ps1):
#   REMOTE_VENV=hw3 bash deploy/remote_train.sh configs/P1.yaml 0
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server

CONFIG="${1:?usage: remote_train.sh <config.yaml> [seed]}"
SEED="${2:-0}"
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

name="$(basename "$CONFIG" .yaml)"
task="$(awk '/^task:/{print $2; exit}' "$CONFIG")"
algo="$(awk '/^algo:/{print $2; exit}' "$CONFIG")"
case "${task}_${algo}" in
  pong_dqn)    script="pong/train_pong_dqn.py" ;;
  pong_ppo)    script="pong/train_pong_ppo.py" ;;
  vizdoom_ppo) script="vizdoom/train_vizdoom_ppo.py" ;;
  vizdoom_dqn) script="vizdoom/train_vizdoom_dqn.py" ;;
  *) echo "ERROR: cannot map task='$task' algo='$algo' from $CONFIG" >&2; exit 1 ;;
esac

log="logs/${name}_s${SEED}.log"
pidf="logs/${name}_s${SEED}.pid"

# nohup + redirect + </dev/null detaches the job so it keeps running after the
# SSH session closes. -u keeps Python output unbuffered for live tailing.
nohup bash -c "source deploy/_activate.sh && activate_env && exec python -u '$script' --config '$CONFIG' --seed '$SEED'" \
  > "$log" 2>&1 < /dev/null &
echo $! > "$pidf"

echo "STARTED $name seed=$SEED  pid=$(cat "$pidf")  script=$script"
echo "  log: $log    (track:  tail -f $log)"
