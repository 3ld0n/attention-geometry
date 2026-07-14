"""
exp-084 — Modal cloud launcher (PCFG discriminator).

Runs ONE training run on C-PCFG_full.bin using the IDENTICAL protocol to exp-062
(same train.py and measure.py, same architecture, same optimizer, same schedule).

Pre-registration of record: notes.md (committed 2026-07-09).
Amendment 2026-07-13: β̂ matching not achievable; run proceeds with β̂=2.97.

Volume: exp084-data  (separate from exp062-data to keep experiments clean)

Usage (from repo root):
  # Upload corpus and scripts:
  .venv/bin/python -m modal run research/physics/experiments/\
exp-084_pcfg_discriminator/modal_exp084.py --phase upload

  # Train:
  .venv/bin/python -m modal run --detach research/physics/experiments/\
exp-084_pcfg_discriminator/modal_exp084.py --phase train

  # Measure:
  .venv/bin/python -m modal run research/physics/experiments/\
exp-084_pcfg_discriminator/modal_exp084.py --phase measure

  # All (upload → train → measure):
  .venv/bin/python -m modal run --detach research/physics/experiments/\
exp-084_pcfg_discriminator/modal_exp084.py --phase all

Phases: upload | train | measure | all
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# ─── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"

app = modal.App("exp084-pcfg-discriminator")
vol = modal.Volume.from_name("exp084-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy==2.4.6",
    "scipy==1.17.1",
    "torch==2.12.0",
    "transformers==5.8.1",
)

# Single pre-registered run (identical init/data seeds to exp-062 C-PL40 for comparability)
RUN_NAME  = "run_CPCFG"
INIT_SEED = 1000
DATA_SEED = 2000
CORPUS    = "C-PCFG"


# ─── upload ───────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "all") -> None:
    """Orchestrator: upload → train → measure."""

    if phase in ("upload", "all"):
        _upload_to_volume()
        print("Upload complete.")
        if phase == "upload":
            return

    if phase in ("train", "all"):
        print(f"Launching training run {RUN_NAME} ...")
        result = train_run.remote(CORPUS, RUN_NAME, INIT_SEED, DATA_SEED)
        print("TRAINED:", result)

    if phase in ("measure", "all"):
        print(f"Measuring {RUN_NAME} at step 2000 ...")
        out = measure_run.remote(RUN_NAME, step=2000)
        print(out)
        # Download results with CLI:
        #   modal volume get exp084-data measurements/run_CPCFG_final.json \
        #     research/physics/experiments/exp-084_pcfg_discriminator/measurements/
        meas_dir = SCRIPT_DIR / "measurements"
        meas_dir.mkdir(exist_ok=True)
        tag = f"{RUN_NAME}_final"
        try:
            import subprocess as _sp
            _sp.run(
                ["modal", "volume", "get", "exp084-data",
                 f"measurements/{tag}.json",
                 str(meas_dir / f"{tag}.json")],
                check=True
            )
            print(f"Downloaded measurements/{tag}.json → {meas_dir}")
        except Exception as e:
            print(f"Auto-download failed ({e}); run manually:")
            print(f"  modal volume get exp084-data measurements/{tag}.json "
                  f"research/physics/experiments/exp-084_pcfg_discriminator/measurements/")

    print("EXP084_DONE")


def _upload_to_volume() -> None:
    """Upload corpus, alphabet, and scripts to the Modal volume."""
    corpus_bin  = SCRIPT_DIR / "corpora" / "C-PCFG_full.bin"
    alphabet    = EXP062_DIR / "alphabet.json"
    train_py    = EXP062_DIR / "train.py"
    measure_py  = EXP062_DIR / "measure.py"

    for p in [corpus_bin, alphabet, train_py, measure_py]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    print(f"Uploading {corpus_bin.name} ({corpus_bin.stat().st_size/1e9:.2f} GB) ...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(corpus_bin), f"corpora/{corpus_bin.name}")
        batch.put_file(str(alphabet),   "alphabet.json")
        batch.put_file(str(train_py),   "train.py")
        batch.put_file(str(measure_py), "measure.py")
    print("Volume upload complete.")




# ─── remote functions ─────────────────────────────────────────────────────────

@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=8 * 3600)
def train_run(corpus: str, run_name: str, init_seed: int, data_seed: int) -> str:
    """Train one run. Identical to exp-062 train_run — same protocol."""
    src = Path(f"/vol/corpora/{corpus}_full.bin")
    dst = Path(f"/tmp/{corpus}_full.bin")
    t0 = time.time()
    shutil.copyfile(src, dst)
    print(f"corpus copied in {time.time()-t0:.0f}s", flush=True)

    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(
        [sys.executable, "/vol/train.py", str(dst), run_name,
         "--init-seed", str(init_seed), "--data-seed", str(data_seed),
         "--micro-batch", "64", "--device", "cuda"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )

    def _commit_loop() -> None:
        while proc.poll() is None:
            time.sleep(300)
            try:
                vol.commit()
            except Exception as e:
                print(f"[commit] {e}", flush=True)

    threading.Thread(target=_commit_loop, daemon=True).start()
    for line in proc.stdout:
        print(f"[{run_name}] {line}", end="", flush=True)
    rc = proc.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"{run_name} exited {rc}")
    return run_name


@app.function(image=image, volumes={"/vol": vol}, gpu="A10G", timeout=2 * 3600)
def measure_run(run_name: str, step: int = 2000) -> str:
    """Measure conformal heads at checkpoint step. Identical protocol to exp-062."""
    vol.reload()
    tag = f"{run_name}_final"
    cmd = [sys.executable, "/vol/measure.py",
           f"/vol/runs/{run_name}/step_{step}", tag,
           "--alphabet", "/vol/alphabet.json"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout, flush=True)
    if out.returncode != 0:
        print(out.stderr, flush=True)
        raise RuntimeError(f"measure {tag} exited {out.returncode}")
    vol.commit()
    return out.stdout.strip()
