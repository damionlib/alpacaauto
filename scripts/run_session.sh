#!/usr/bin/env bash
# Run the agent loop for one regular trading session (paper only), then exit.
#
# Started by launchd before the open on weekdays (see
# scripts/com.local.trading-agent.session.plist). It exits immediately on market
# holidays, when another agent loop is already running, or when the broker mode
# is not paper; it stops 20 minutes after the actual close (early closes too).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/session-$(date +%Y-%m-%d).log"
exec >>"$LOG" 2>&1
echo "$(date '+%F %T') runner start"

if pgrep -f "trading-agent loop" >/dev/null; then
  echo "another trading-agent loop is already running; exiting"
  exit 0
fi
if ! grep -Eq '^mode *= *"paper"' config/settings.toml; then
  echo "broker mode is not paper; refusing to run unattended"
  exit 2
fi

WINDOW=$(.venv/bin/trading-agent session-window) || { echo "broker clock check failed"; exit 1; }
echo "session window: $WINDOW"
STOP=$(printf '%s' "$WINDOW" | .venv/bin/python -c 'import json, sys; print(json.load(sys.stdin)["stop_epoch"])')
if [ "$STOP" -le 0 ]; then
  echo "not a trading day; exiting"
  exit 0
fi

# Keep the Mac from idle-sleeping while this runner is alive.
caffeinate -i -w $$ &

restarts=0
while [ "$(date +%s)" -lt "$STOP" ]; do
  COLUMNS=200 .venv/bin/trading-agent loop &
  PID=$!
  while [ "$(date +%s)" -lt "$STOP" ] && kill -0 "$PID" 2>/dev/null; do
    sleep 20
  done
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    wait "$PID" 2>/dev/null
    break
  fi
  restarts=$((restarts + 1))
  echo "$(date '+%F %T') agent loop exited unexpectedly (restart $restarts)"
  if [ "$restarts" -ge 5 ]; then
    echo "too many restarts; giving up for today"
    break
  fi
  sleep 30
done
echo "$(date '+%F %T') runner stop"
