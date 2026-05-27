#!/usr/bin/env bash
# Download model-weight trees only (~24.5 GB per arm), per Apple ml-sigmoid-attention notebook.
# Full checkpoint is ~73 GB; do not pull learner state unless needed.
#
# Requires: gsutil, gcloud auth. Confirm ~50 GB total before running.
set -euo pipefail

ROOT="${ARIEL_SIGMOID_CKPT_ROOT:-$HOME/ariel-data/apple-gala-7b}"
mkdir -p "$ROOT"

SIGMOID="gs://axlearn-public/experiments/gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/checkpoints/step_00250000"
SOFTMAX="gs://axlearn-public/experiments/gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/checkpoints/step_00250000"

download_arm() {
  local uri="$1"
  local dest="$2"
  echo "=== $dest ==="
  mkdir -p "$dest"
  gsutil -m cp -r "${uri}/index" "$dest/"
  gsutil -m cp -r "${uri}/gda/" "$dest/gda/"
  for i in $(seq 0 31); do
    if gsutil ls "${uri}/tf_${i}/" &>/dev/null; then
      gsutil -m cp -r "${uri}/tf_${i}/" "$dest/tf_${i}/"
    fi
  done
}

download_arm "$SIGMOID" "$ROOT/sigmoid-step_00250000"
download_arm "$SOFTMAX" "$ROOT/softmax-step_00250000"

echo "Done. Weights under $ROOT"
