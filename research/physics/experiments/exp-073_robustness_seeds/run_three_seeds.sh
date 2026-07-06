#!/bin/bash
# exp-073 — run the three new seeds sequentially on MPS.
# Gate (seed 42) PASSED 2026-07-04 (exact match vs CUDA fp32; see notes.md).
# Launched detached (start_new_session) so it survives the launching session —
# two previous attempts were SIGKILLed when the Cursor shell session closed.
cd "$(dirname "$0")"
PY=/Users/ariel/ariel/.venv/bin/python3
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> overnight_log.txt; }

log "detached three-seed run starting (pid $$, sid $(ps -o sess= -p $$ 2>/dev/null | tr -d ' '))"
for seed in 123 2024 99; do
  log "=== seed $seed starting ==="
  t0=$SECONDS
  "$PY" run_local_mps.py --seed "$seed" >> overnight_log.txt 2>&1
  rc=$?
  log "=== seed $seed exited rc=$rc in $(( (SECONDS - t0) / 60 )) min ==="
done
log "all three seeds done — review then aggregate.py"
