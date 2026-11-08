#!/usr/bin/env bash
set -euo pipefail

# The official-quota gate supplies the outer bound for this one attempt.
outer_seconds="$1"
shift
if ! [[ "$outer_seconds" =~ ^[0-9]+$ ]] || (( outer_seconds < 900 || outer_seconds > 28800 )); then
  echo "Invalid bounded wall budget" >&2
  exit 2
fi
internal_seconds=$((outer_seconds - 300))
exec timeout --signal=TERM --kill-after=30s "${outer_seconds}s" \
  python -u runner.py --wall-seconds "$internal_seconds" "$@"
