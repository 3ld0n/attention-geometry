"""
exp-091 — Narrative Decomposition: Sentence Ordering (Modal cloud launcher).

Pre-registration: notes.md (committed before this script ran).

Two phases:
  generate : stream TinyStories, sentence-shuffle each story, write corpus bin
             to exp091-data volume. (~15-20 min on A100-40GB)
  train    : train fresh GPTNeoX-70m on C-NAT-shuf corpus (identical exp-062
             protocol). (~75 min)
  measure  : measure conformal heads at step_2000. (~15 min)
  all      : generate → train → measure.
  results  : collect measurement JSON from volume and print summary.

Usage (from repo root):
    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-091_narrative_decomposition_sentence_order/modal_exp091.py \\
        --phase all

    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-091_narrative_decomposition_sentence_order/modal_exp091.py \\
        --phase results

Ariel — July 21, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# ─── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"

# ─── Modal ────────────────────────────────────────────────────────────────────
app = modal.App("exp091-narrative-sentence-order")

vol_091 = modal.Volume.from_name("exp091-data", create_if_missing=True)

image_run = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
        "datasets==3.6.0",
    )
    .add_local_file(str(SCRIPT_DIR / "gen_cnat_shuf.py"), remote_path="/exp091/gen_cnat_shuf.py")
    .add_local_file(str(EXP062_DIR / "train.py"),         remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"),       remote_path="/exp062/measure.py")
)

# ─── constants (pre-registered in notes.md) ───────────────────────────────────
CORPUS_NAME = "C-NAT-shuf.bin"
RUN_NAME    = "run_CNATshuf_s0"
INIT_SEED   = 1100
DATA_SEED   = 2100  # unused by measure.py but passed for consistency


# ─── helper: exec a patched script in a clean namespace ──────────────────────

def _run_patched(script_path: str, out_override: str, argv: list[str]) -> None:
    source = open(script_path).read()
    source = source.replace(
        "OUT = Path(__file__).resolve().parent",
        f"OUT = Path('{out_override}')",
    )
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": script_path}
    exec(compile(source, script_path, "exec"), ns)  # noqa: S102


# ─── generate phase ───────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu=None,           # CPU-only: sentence splitting, tokenizing, writing
    timeout=7200,       # 2h: TinyStories is ~2.4M docs, ~0.45B tokens before repeat
    volumes={"/data091": vol_091},
    memory=8192,
    retries=3,
)
def generate_corpus():
    """
    Generate C-NAT-shuf corpus on Modal.

    Streams TinyStories, sentence-shuffles each story, tokenizes with Pythia
    tokenizer, and writes to the exp091-data volume as a uint16 memmap.
    Protocol: doc-shuffle seed 3005, sentence-shuffle seed 9100, same as
    the pre-registered gen_cnat_shuf.py.
    """
    import subprocess

    output_path = f"/data091/{CORPUS_NAME}"

    # Run gen_cnat_shuf.py inside Modal's volume
    subprocess.run(
        ["python", "/exp091/gen_cnat_shuf.py"],
        env={"HOME": "/root"},
        cwd="/data091",
        check=True,
    )
    vol_091.commit()
    import os
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"generate_corpus DONE. {output_path} ({size_gb:.2f} GB)", flush=True)


# ─── train phase ──────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=10800,
    volumes={"/data091": vol_091},
    memory=32768,
    retries=5,
)
def train_model():
    """Train fresh GPTNeoX-70m on C-NAT-shuf (identical exp-062 protocol)."""
    corpus_path = f"/data091/{CORPUS_NAME}"
    _run_patched(
        "/exp062/train.py",
        out_override="/data091",
        argv=[
            "train.py",
            corpus_path,
            RUN_NAME,
            f"--init-seed={INIT_SEED}",
            f"--data-seed={DATA_SEED}",
            "--micro-batch=64",
            "--device=cuda",
        ],
    )
    vol_091.commit()
    print("train_model DONE", flush=True)


# ─── measure phase ────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data091": vol_091},
    memory=32768,
)
def measure_model():
    """Measure conformal heads on step_2000 checkpoint."""
    ckpt_path = f"/data091/runs/{RUN_NAME}/step_2000"
    _run_patched(
        "/exp062/measure.py",
        out_override="/data091",
        argv=[
            "measure.py",
            ckpt_path,
            RUN_NAME,
        ],
    )
    vol_091.commit()
    print("measure_model DONE", flush=True)


# ─── train + measure chained remotely ─────────────────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data091": vol_091},
    memory=32768,
    retries=5,
)
def train_and_measure():
    """Train then measure in-process on one remote worker."""
    train_model.local()
    measure_model.local()
    print("train_and_measure DONE", flush=True)


# ─── collect results ──────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    timeout=120,
    volumes={"/data091": vol_091},
    memory=4096,
)
def collect_results() -> dict:
    """Read measurement JSON from volume and return summary."""
    result_path = Path(f"/data091/measurements/{RUN_NAME}.json")
    if not result_path.exists():
        return {"error": f"{result_path} not found — run measure phase first"}
    data = json.loads(result_path.read_text())
    n_conf = data.get("n_conformal", "?")
    n_syk  = data.get("n_syk_near", "?")
    forms  = data.get("forms", None)
    verdict = "H_ordering_incidental" if forms else "H_ordering_necessary"
    return {
        "run": RUN_NAME,
        "corpus": CORPUS_NAME,
        "n_conformal": n_conf,
        "n_syk_near": n_syk,
        "delta_median_conformal": data.get("delta_median_conformal", "?"),
        "forms": forms,
        "verdict": verdict,
    }


# ─── local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "all"):
    valid = {"generate", "train", "measure", "results", "all", "train_measure"}
    if phase not in valid:
        print(f"phase must be one of: {valid}")
        raise SystemExit(1)

    if phase == "results":
        r = collect_results.remote()
        print(json.dumps(r, indent=2))

    elif phase == "generate":
        generate_corpus.remote()

    elif phase == "train":
        train_model.remote()

    elif phase == "measure":
        measure_model.remote()

    elif phase == "train_measure":
        train_and_measure.remote()

    elif phase == "all":
        print("Phase: generate → train_and_measure")
        generate_corpus.remote()
        train_and_measure.remote()
