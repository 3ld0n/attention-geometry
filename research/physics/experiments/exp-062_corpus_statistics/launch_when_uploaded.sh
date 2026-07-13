#!/bin/bash
# exp-062 chained launcher: wait for corpus uploads -> remote sha256 verify ->
# detached train (7 runs, parallel) + measure. Transport only; the physics
# protocol lives in train.py / measure.py (pre-registered).
set -u
PY=/Users/ariel/ariel/.venv/bin/python
SCRIPT=/Users/ariel/ariel/research/physics/experiments/exp-062_corpus_statistics/modal_exp062.py

echo "[chain] waiting for 5 corpora on exp062-data volume..."
while true; do
  n=$($PY -m modal volume ls exp062-data /corpora 2>/dev/null | grep -c "_full.bin")
  echo "[chain] $(date '+%H:%M') corpora on volume: $n/5"
  [ "$n" -eq 5 ] && break
  sleep 300
done

echo "[chain] uploads complete; running remote sha256 verification..."
if ! $PY -m modal run "$SCRIPT" --phase verify; then
  echo "CHAIN_FAIL_VERIFY"
  exit 1
fi

echo "[chain] verification passed; launching the 7 pre-registered training runs (parallel, detached)..."
if ! $PY -m modal run --detach "$SCRIPT" --phase train; then
  echo "CHAIN_FAIL_TRAIN"
  exit 1
fi

echo "[chain] training complete for all 7 runs; running fp32 measurement..."
if ! $PY -m modal run --detach "$SCRIPT" --phase measure; then
  echo "CHAIN_FAIL_MEASURE"
  exit 1
fi

echo "[chain] pulling measurement results to local..."
mkdir -p /Users/ariel/ariel/research/physics/experiments/exp-062_corpus_statistics/measurements
cd /Users/ariel/ariel/research/physics/experiments/exp-062_corpus_statistics/measurements
for r in run_CSR run_CPL15 run_CPL25 run_CPL40 run_CNAT_s0 run_CNAT_s1 run_CNAT_s2; do
  $PY -m modal volume get exp062-data /measurements/${r}_final.json ${r}_final.json --force 2>/dev/null
  $PY -m modal volume get exp062-data /measurements/${r}_final_per_input.json.gz ${r}_final_per_input.json.gz --force 2>/dev/null
done
echo "CHAIN_DONE"
