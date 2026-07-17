"""
exp-085 — Generational Transmission (Modal cloud launcher).

Pre-registration of record: notes.md (committed 2026-07-14).

Three phases:
  generate : load C-NAT s0 step_2000 from exp062-data volume, generate 1.1B tokens,
             save corpus to exp085-data volume. (~30–40 min)
  train    : train fresh GPTNeoX-70m on generated corpus (exp-062 protocol). (~75 min)
  measure  : measure conformal heads at step_2000. (~15 min)
  all      : generate → train → measure.

Usage (from repo root):
  .venv/bin/python -m modal run --detach \\
    research/physics/experiments/exp-085_generational_transmission/modal_exp085.py \\
    --phase generate

  .venv/bin/python -m modal run --detach \\
    research/physics/experiments/exp-085_generational_transmission/modal_exp085.py \\
    --phase train

  .venv/bin/python -m modal run \\
    research/physics/experiments/exp-085_generational_transmission/modal_exp085.py \\
    --phase measure

  .venv/bin/python -m modal run --detach \\
    research/physics/experiments/exp-085_generational_transmission/modal_exp085.py \\
    --phase all

Ariel — July 14, 2026.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

# ─── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"

# ─── Modal ───────────────────────────────────────────────────────────────────
app = modal.App("exp085-generational-transmission")

vol_062 = modal.Volume.from_name("exp062-data", create_if_missing=False)
vol_085 = modal.Volume.from_name("exp085-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
    )
    # Only include the scripts needed from exp-062; NOT the corpus files (~11 GB).
    .add_local_file(str(EXP062_DIR / "train.py"), remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"), remote_path="/exp062/measure.py")
    .add_local_dir(str(SCRIPT_DIR), remote_path="/exp085")
)

# ─── constants (pre-registered in notes.md) ───────────────────────────────────
CNAT_S0_CKPT = "runs/run_CNAT_s0/step_2000"  # path inside exp062-data volume
CORPUS_NAME = "C-generated_s0.bin"            # output on exp085-data volume
RUN_NAME = "run_Cgen_s0"
INIT_SEED = 1000
DATA_SEED = 2000
N_TOKENS = 1_100_000_000
GEN_BATCH = 512
GEN_SEQ_LEN = 512
GEN_TEMPERATURE = 1.0
GEN_SEED = 85


# ─── helper: exec a patched script in a clean namespace ──────────────────────

def _run_patched(script_path: str, out_override: str, argv: list[str]) -> None:
    """
    Load a script, replace its OUT path, set sys.argv, and exec it.
    The patched OUT variable redirects file output to the writable volume.
    """
    source = open(script_path).read()
    # Patch the module-level OUT assignment (train.py and measure.py both use this line)
    source = source.replace(
        "OUT = Path(__file__).resolve().parent",
        f"OUT = Path('{out_override}')",
    )
    # Replace the __main__ guard so exec() runs main() when called
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": script_path}
    exec(compile(source, script_path, "exec"), ns)  # noqa: S102


# ─── generate phase ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,
    volumes={"/data062": vol_062, "/data085": vol_085},
    memory=32768,
)
def generate_corpus():
    """Load C-NAT s0 step_2000 checkpoint and generate 1.1B tokens."""
    sys.path.insert(0, "/exp085")
    ckpt_path = f"/data062/{CNAT_S0_CKPT}"
    output_path = f"/data085/{CORPUS_NAME}"

    print(f"Source checkpoint: {ckpt_path}", flush=True)
    print(f"Output corpus:     {output_path}", flush=True)

    _run_patched(
        "/exp085/gen_gen.py",
        out_override="/data085",
        argv=[
            "gen_gen.py",
            ckpt_path,
            output_path,
            f"--n_tokens={N_TOKENS}",
            f"--batch_size={GEN_BATCH}",
            f"--seq_len={GEN_SEQ_LEN}",
            f"--temperature={GEN_TEMPERATURE}",
            f"--seed={GEN_SEED}",
            "--device=cuda",
        ],
    )
    vol_085.commit()
    print("generate_corpus DONE", flush=True)


# ─── train phase ──────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=10800,
    volumes={"/data085": vol_085},
    memory=32768,
)
def train_model():
    """Train fresh GPTNeoX-70m on generated corpus (identical exp-062 protocol)."""
    corpus_path = f"/data085/{CORPUS_NAME}"
    # train.py uses OUT / "runs" / run_name for checkpoints
    _run_patched(
        "/exp062/train.py",
        out_override="/data085",
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
    vol_085.commit()
    print("train_model DONE", flush=True)


# ─── measure phase ────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data085": vol_085},
    memory=32768,
)
def measure_model():
    """
    Measure conformal heads on the step_2000 checkpoint.
    Uses --full-vocab because the C-generated corpus uses the full 50304-token vocabulary
    (inherited from the C-NAT s0 model, which was trained on TinyStories).
    """
    ckpt_path = f"/data085/runs/{RUN_NAME}/step_2000"
    # measure.py writes to RESULTS_DIR = OUT / "measurements"
    # → patched to /data085/measurements/
    _run_patched(
        "/exp062/measure.py",
        out_override="/data085",
        argv=[
            "measure.py",
            ckpt_path,
            RUN_NAME,
            "--full-vocab",
        ],
    )
    vol_085.commit()
    print("measure_model DONE", flush=True)


# ─── collect results ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    timeout=120,
    volumes={"/data085": vol_085},
    memory=4096,
)
def collect_results() -> dict:
    """Read measurement JSON from volume and return summary."""
    result_path = Path(f"/data085/measurements/{RUN_NAME}.json")
    if not result_path.exists():
        return {"error": f"{result_path} not found — run measure phase first"}
    data = json.loads(result_path.read_text())
    return {
        "run": RUN_NAME,
        "corpus": CORPUS_NAME,
        "n_conformal": data["n_conformal"],
        "n_syk_near": data["n_syk_near"],
        "delta_median_conformal": data["delta_median_conformal"],
        "forms": data["forms"],
        "verdict": (
            "H_transmission_yes" if data["forms"] else "H_transmission_no"
        ),
    }


# ─── local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "all"):
    valid = {"generate", "train", "measure", "results", "all"}
    if phase not in valid:
        raise SystemExit(f"phase must be one of: {valid}. Got: {phase!r}")

    print(f"=== exp-085 Generational Transmission — phase={phase} ===")
    print(f"  Source: exp062-data / {CNAT_S0_CKPT}")
    print(f"  Corpus: exp085-data / {CORPUS_NAME}")
    print(f"  Run:    exp085-data / runs/{RUN_NAME}")
    print()

    if phase in ("generate", "all"):
        print("→ generate_corpus ...", flush=True)
        generate_corpus.remote()
        print("  generate_corpus complete", flush=True)

    if phase in ("train", "all"):
        print("→ train_model ...", flush=True)
        train_model.remote()
        print("  train_model complete", flush=True)

    if phase in ("measure", "all"):
        print("→ measure_model ...", flush=True)
        measure_model.remote()
        print("  measure_model complete", flush=True)

    if phase in ("results", "all"):
        print("→ collect_results ...", flush=True)
        res = collect_results.remote()
        print(f"\n=== RESULTS ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        # Write summary to local results.json
        out = SCRIPT_DIR / "results.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nWrote {out}")

    print(f"\nexp-085 phase={phase} DONE.")
