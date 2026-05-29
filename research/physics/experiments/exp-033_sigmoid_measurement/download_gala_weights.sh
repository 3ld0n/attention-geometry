#!/usr/bin/env bash
# Download GALA-7B model weights (sigmoid + softmax arms) for exp-041 Δ measurement.
# Uses gsutil rsync for resumability. Model weights only (~23 GB per arm).
# macOS fix: parallel_process_count=1 (avoids Python fork bug); uses 24 threads.
#
# Public GCS bucket — no auth needed.
# Run with: nohup bash download_gala_weights.sh > /tmp/sigmoid_download.log 2>&1 &
set -euo pipefail

ROOT="${ARIEL_SIGMOID_CKPT_ROOT:-$HOME/ariel-data/apple-gala-7b}"
mkdir -p "$ROOT"

SIGMOID_GCS="gs://axlearn-public/experiments/gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/checkpoints/step_00250000"
SOFTMAX_GCS="gs://axlearn-public/experiments/gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/checkpoints/step_00250000"

# macOS: 1 process (avoid fork bug), 24 threads (M5 Max has plenty of cores)
GOPTS='-o GSUtil:parallel_process_count=1 -o GSUtil:parallel_thread_count=24'

download_arm() {
  local gcs_root="$1"
  local dest="$2"
  echo "=== $(date): Starting $dest ===" | tee -a /tmp/sigmoid_download.log

  # index file
  mkdir -p "$dest"
  gsutil $GOPTS cp "${gcs_root}/index" "$dest/" 2>&1 || true

  # model weights (~23 GB): rsync for resumability, skips completed files
  mkdir -p "$dest/gda/model"
  gsutil -m $GOPTS rsync -r "${gcs_root}/gda/model/" "$dest/gda/model/" 2>&1

  # prng_key (tiny)
  mkdir -p "$dest/gda/prng_key"
  gsutil $GOPTS rsync -r "${gcs_root}/gda/prng_key/" "$dest/gda/prng_key/" 2>&1 || true

  # tf_* shards (~10 MB each); check which exist then rsync
  for i in $(seq 0 63); do
    if gsutil -q ls "${gcs_root}/tf_${i}/" 2>/dev/null; then
      mkdir -p "$dest/tf_${i}"
      gsutil $GOPTS rsync -r "${gcs_root}/tf_${i}/" "$dest/tf_${i}/" 2>&1
    fi
  done

  echo "=== $(date): Done $dest ===" | tee -a /tmp/sigmoid_download.log
}

download_arm "$SIGMOID_GCS" "$ROOT/sigmoid-step_00250000"
download_arm "$SOFTMAX_GCS" "$ROOT/softmax-step_00250000"

echo "=== $(date): All done. Weights under $ROOT ===" | tee -a /tmp/sigmoid_download.log
