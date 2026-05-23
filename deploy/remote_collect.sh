#!/usr/bin/env bash
# Bundle the lightweight, report-relevant artifacts into one tarball for scp
# back to the laptop (used by deploy/fetch.ps1 -Lite). Keeps models/buffers/
# tensorboard out so the transfer stays small.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
out="results_lite.tgz"

mapfile -t files < <(find results -type f \
  \( -name metrics.csv -o -name curve.png -o -name eval.json \
     -o -name config.yaml -o -name gameplay.gif \) 2>/dev/null || true)

if [ "${#files[@]}" -gt 0 ]; then
  tar -czf "$out" "${files[@]}" logs 2>/dev/null || tar -czf "$out" "${files[@]}"
else
  # No results yet -- still ship the logs so progress can be inspected.
  tar -czf "$out" logs 2>/dev/null || { echo "nothing to collect" >&2; exit 1; }
fi
echo "$out"
