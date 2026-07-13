"""
exp-062 — Modal cloud launcher (Phase 1.1, pre-registered).

Wraps the pre-registered train.py / measure.py VERBATIM (both uploaded to the
exp062-data volume alongside the five full corpora + alphabet.json). Nothing
in the physics protocol lives in this file — it is transport only.

Pre-registration of record: notes/2026-06-11_corpus_statistics_preregistration.md.
Runbook: LAUNCH.md. Library versions pinned to the exact versions used to
generate the corpora locally (numpy 2.4.6 / scipy 1.17.1 / torch 2.12.0 /
transformers 5.8.1) so the data-seed batch stream is identical.

Usage (from repo root):
  .venv/bin/python -m modal run --detach research/physics/experiments/\
exp-062_corpus_statistics/modal_exp062.py --phase verify
  .venv/bin/python -m modal run --detach ...::main --phase all

Phases: verify | train | measure | all
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

app = modal.App("exp062-corpus-statistics")
vol = modal.Volume.from_name("exp062-data")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy==2.4.6",
    "scipy==1.17.1",
    "torch==2.12.0",
    "transformers==5.8.1",
)

# The 7 pre-registered runs (prereg §3 / LAUNCH.md step 2).
RUNS = [
    ("C-SR", "run_CSR", 1000, 2000),
    ("C-PL15", "run_CPL15", 1000, 2000),
    ("C-PL25", "run_CPL25", 1000, 2000),
    ("C-PL40", "run_CPL40", 1000, 2000),
    ("C-NAT", "run_CNAT_s0", 1000, 2000),
    ("C-NAT", "run_CNAT_s1", 1001, 2001),
    ("C-NAT", "run_CNAT_s2", 1002, 2002),
]


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@app.function(image=image, volumes={"/vol": vol}, timeout=3600)
def verify_corpora() -> dict:
    """Check uploaded corpora + alphabet against the local sha256 manifest."""
    manifest = {}
    for line in Path("/vol/corpora_sha256.txt").read_text().splitlines():
        digest, name = line.split()
        manifest[name] = digest  # names are volume-relative (corpora/X.bin, alphabet.json)
    results = {}
    for name, expected in manifest.items():
        p = Path("/vol") / name
        if not p.exists():
            results[name] = "MISSING"
            continue
        got = _sha256(p)
        results[name] = "OK" if got == expected else f"MISMATCH ({got})"
        print(f"{name}: {results[name]}", flush=True)
    return results


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=8 * 3600)
def train_run(corpus: str, run_name: str, init_seed: int, data_seed: int) -> str:
    """One pre-registered training run. Checkpoints land on the volume at
    /vol/runs/<run_name>; committed every 5 min so a preempted run resumes."""
    # Copy corpus to container-local disk: memmap over the network mount is slow.
    src = Path(f"/vol/corpora/{corpus}_full.bin")
    dst = Path(f"/tmp/{corpus}_full.bin")
    t0 = time.time()
    shutil.copyfile(src, dst)
    print(f"corpus copied to local disk in {time.time()-t0:.0f}s", flush=True)

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
            except Exception as e:  # commit races are non-fatal; next loop retries
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
def measure_run(run_name: str, step: int = 2000, full_vocab: bool = False,
                tag: str | None = None) -> str:
    """Frozen-protocol measurement (fp32) on a checkpoint from the volume."""
    vol.reload()
    tag = tag or f"{run_name}_final"
    cmd = [sys.executable, "/vol/measure.py",
           f"/vol/runs/{run_name}/step_{step}", tag]
    if full_vocab:
        cmd.append("--full-vocab")
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout, flush=True)
    if out.returncode != 0:
        print(out.stderr, flush=True)
        raise RuntimeError(f"measure {tag} exited {out.returncode}")
    vol.commit()
    return out.stdout.strip()


@app.local_entrypoint()
def main(phase: str = "all", runs: str = "") -> None:
    """phase: verify | train | measure | all. runs: optional comma-separated
    run-name filter (e.g. 'run_CSR') for a staged launch."""
    selected = [r for r in RUNS if not runs or r[1] in runs.split(",")]

    if phase in ("verify", "all"):
        results = verify_corpora.remote()
        bad = {k: v for k, v in results.items() if v != "OK"}
        if bad:
            raise SystemExit(f"corpus verification failed: {bad}")
        print("all corpora verified OK")
        if phase == "verify":
            return

    if phase in ("train", "all"):
        handles = [train_run.spawn(c, r, i, d) for c, r, i, d in selected]
        for h in handles:
            print("TRAINED:", h.get())

    if phase in ("measure", "all"):
        for corpus, run_name, _, _ in selected:
            print(measure_run.remote(run_name, full_vocab=(corpus == "C-NAT")))
    print("EXP062_CLOUD_DONE")
