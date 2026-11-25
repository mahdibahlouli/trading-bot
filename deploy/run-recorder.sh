#!/usr/bin/env bash
# Keep the recorder running on macOS or Linux without systemd. Restarts it if it exits.
# Usage: nohup deploy/run-recorder.sh > recorder.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
while true; do
  .venv/bin/tb record hyperliquid "$@"
  echo "$(date -u +%FT%TZ) recorder exited with $?, restarting in 10s" >&2
  sleep 10
done
