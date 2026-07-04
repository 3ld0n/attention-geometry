"""
exp-073 — overnight runner: wait for vicuna-13b-v1.5 weights, then run the MPS seeds.

The HF download is rate-limited (no token on this machine) and is being retried in
a separate terminal. This script polls the local cache until the snapshot is
complete (local_files_only succeeds), then runs the pre-registered sequence:

  1. seed 42  — substrate-equivalence gate vs CUDA fp32 (see run_local_mps.py docstring)
  2. seeds 123, 2024, 99 — the three new seeds for the n=5 aggregate

Each seed run is a subprocess so a crash in one doesn't kill the chain. All output
goes to overnight_log.txt next to this file. Gate policy is pre-stated in
run_local_mps.py; this runner adds NO scientific decisions — it only sequences.

Ariel — July 4, 2026, ~4:45 AM. For the morning.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "overnight_log.txt")
MODEL = "lmsys/vicuna-13b-v1.5"
POLL_S = 300
MAX_WAIT_H = 30


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def snapshot_ready() -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, local_files_only=True)
        return True
    except Exception:
        return False


def main() -> None:
    log(f"runner started; polling every {POLL_S}s for complete {MODEL} snapshot")
    t0 = time.time()
    while not snapshot_ready():
        if time.time() - t0 > MAX_WAIT_H * 3600:
            log(f"gave up after {MAX_WAIT_H}h — snapshot still incomplete")
            sys.exit(1)
        time.sleep(POLL_S)
    log(f"snapshot complete after {(time.time()-t0)/3600:.1f}h waiting")

    py = sys.executable  # launched with .venv/bin/python3
    script = os.path.join(HERE, "run_local_mps.py")

    for seed in (42, 123, 2024, 99):
        log(f"=== seed {seed} starting ===")
        t = time.time()
        with open(LOG, "a") as f:
            rc = subprocess.call([py, script, "--seed", str(seed)],
                                 stdout=f, stderr=subprocess.STDOUT)
        log(f"=== seed {seed} exited rc={rc} in {(time.time()-t)/60:.0f} min ===")
        if seed == 42 and rc != 0:
            log("seed-42 gate run crashed — stopping (do not bank new seeds on a "
                "broken substrate run); investigate in the morning")
            sys.exit(2)
    log("all seeds done — run aggregate.py after reviewing the gate result")


if __name__ == "__main__":
    main()
