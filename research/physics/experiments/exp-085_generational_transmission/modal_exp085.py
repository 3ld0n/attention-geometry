"""
exp-085 — Generational Transmission (Modal cloud launcher).

Pre-registration of record: notes.md (committed 2026-07-14).

Three phases:
  generate : load C-NAT s0 step_2000 from exp062-data volume, generate 1.1B tokens,
             save corpus to exp085-data volume.
  train    : train fresh GPTNeoX-70m on generated corpus (exp-062 protocol). (~75 min)
  measure  : measure conformal heads at step_2000. (~15 min)
  all      : generate → train → measure.
  results  : collect measurement JSON from volume and print summary.

Generate phase history:
  v1 (2026-07-16): model.generate() loop — OOMed without KV cache, empty corpus.
  v2 (2026-07-17): model.generate() with KV cache — 0.05M tok/s, exceeds timeout.
  v3 (2026-07-17): vLLM (gen_gen_vllm.py) — image built but flashinfer JIT
    failed at runtime: 'Could not find nvcc; /usr/local/cuda doesn't exist'.
    Modal debian_slim has CUDA runtime but not the CUDA dev toolkit (no nvcc).
  v4 (2026-07-18): torch.compile + StaticCache via gen_gen_compile.py — uses
    image_train (same image as train/measure). Measured 0.058M tok/s (batch 50,
    A100-40GB). Prior 0.125M tok/s estimate was incorrect.
  v5 (2026-07-18): Chunked generation with periodic vol.commit() for reliability.
    Modal spot-instance preemption caused two failures (at ~15 min and ~4 min).
    Fix: generate in 100M-token chunks, calling vol_085.commit() between chunks.
    retries=2 added. BATCH_SIZE reduced 2048→512 (less memory during graph capture,
    same throughput since bottleneck is KV-cache bandwidth, not compute).

All three phases now use the same image_train for consistency.

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

  .venv/bin/python -m modal run \\
    research/physics/experiments/exp-085_generational_transmission/modal_exp085.py \\
    --phase results

Ariel — July 14, 2026. Generate redesigns July 17–18, 2026.
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

# ── image for all three phases ────────────────────────────────────────────────
# Identical to the exp-062 image; pinned to the same versions for exact
# protocol match with exp-062 and exp-084.
# Used for generate (gen_gen_compile.py), train, and measure.
#
# vLLM (gen_gen_vllm.py) was tried for the generate phase but failed:
# flashinfer JIT requires nvcc, which is not in Modal's debian_slim image.
# The torch.compile + StaticCache approach (gen_gen_compile.py) works in
# the same image as train/measure.
image_train = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
    )
    .add_local_file(str(SCRIPT_DIR / "gen_gen_compile.py"), remote_path="/exp085/gen_gen_compile.py")
    .add_local_file(str(EXP062_DIR / "train.py"),   remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"), remote_path="/exp062/measure.py")
)

# ─── constants (pre-registered in notes.md) ───────────────────────────────────
CNAT_S0_CKPT = "runs/run_CNAT_s0/step_2000"  # path inside exp062-data volume
CORPUS_NAME = "C-generated_s0.bin"            # output on exp085-data volume
RUN_NAME = "run_Cgen_s0"
INIT_SEED = 1000
DATA_SEED = 2000
N_TOKENS = 1_100_000_000
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
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": script_path}
    exec(compile(source, script_path, "exec"), ns)  # noqa: S102


# ─── generate phase (torch.compile + StaticCache, chunked) ───────────────────
# vLLM was tried but flashinfer JIT failed (no nvcc in debian_slim).
# Fallback: gen_gen_compile.py using transformers model.generate() with
# StaticCache. Throughput: 0.058M tok/s on A100-40GB (batch 50, measured).
#
# v5 redesign (2026-07-18): chunked generation with periodic vol.commit().
# Two Modal spot-instance preemptions caused failures at ~15 min and ~4 min.
# Fix: generate 100M tokens per chunk, commit to volume between chunks.
# With retries=2, a preemption only loses the current chunk (<30 min of work).
# BATCH_SIZE reduced 2048→512 for lower memory during CUDA graph capture.

CHUNK_TOKENS = 100_000_000  # 100M tokens per chunk; ~1724s each on A100-40GB

@app.function(
    image=image_train,
    gpu="A100-40GB",
    timeout=21600,     # 6h: 11 chunks × ~1724s ≈ 18964s, within limit
    volumes={"/data062": vol_062, "/data085": vol_085},
    memory=32768,
    retries=10,        # retry on preemption (resume logic preserves committed progress)
)
def generate_corpus():
    """
    Generate 1.1B tokens from C-NAT s0 step_2000 checkpoint.

    Protocol (pre-registered in notes.md):
      - Temperature 1.0, seq_len 512, seed 85
      - Prompt: 1 token drawn uniformly from vocabulary per sequence
      - Output: uint16 binary, same format as exp-062 corpora
    Uses torch.compile + StaticCache (gen_gen_compile.py, BATCH_SIZE=512).
    Throughput: ~0.058M tok/s steady-state (batch 50, A100-40GB).
    Chunked: generates 100M tokens then commits to volume before continuing.
    """
    from pathlib import Path
    import subprocess

    ckpt_path   = f"/data062/{CNAT_S0_CKPT}"
    output_path = f"/data085/{CORPUS_NAME}"

    print(f"Source checkpoint: {ckpt_path}", flush=True)
    print(f"Output corpus    : {output_path}", flush=True)

    # Chunked loop: generate CHUNK_TOKENS at a time, commit between chunks.
    # gen_gen_compile.py uses --n_tokens as a target: it stops when the output
    # file has >= n_tokens tokens. Resume logic continues from existing data.
    chunk_start = 0
    if Path(output_path).exists():
        chunk_start = Path(output_path).stat().st_size // 2  # existing tokens

    print(f"Starting from {chunk_start:,} tokens already on volume.", flush=True)

    chunk_end = chunk_start
    while chunk_end < N_TOKENS:
        chunk_end = min(chunk_start + CHUNK_TOKENS, N_TOKENS)
        # chunk_start advances to the next boundary after each successful chunk
        next_target = chunk_end

        print(f"\n→ chunk {chunk_start:,} → {next_target:,} tokens", flush=True)
        subprocess.run(
            [
                "python", "/exp085/gen_gen_compile.py",
                ckpt_path,
                output_path,
                f"--n_tokens={next_target}",
                f"--seq_len={GEN_SEQ_LEN}",
                f"--temperature={GEN_TEMPERATURE}",
                f"--seed={GEN_SEED}",
            ],
            check=True,
        )

        vol_085.commit()
        actual_tokens = Path(output_path).stat().st_size // 2
        print(f"  committed: {actual_tokens:,} tokens on volume", flush=True)
        chunk_start = actual_tokens  # advance to actual written position

    print("\ngenerate_corpus DONE", flush=True)


# ─── train phase ──────────────────────────────────────────────────────────────

@app.function(
    image=image_train,
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
    image=image_train,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data085": vol_085},
    memory=32768,
)
def measure_model():
    """
    Measure conformal heads on the step_2000 checkpoint.
    Uses --full-vocab because the C-generated corpus inherits the full 50304-token
    vocabulary from the C-NAT s0 model (trained on TinyStories).
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


# ─── train + measure in one remote function ──────────────────────────────────
# Added 2026-07-20: local wrapper processes don't survive on the launching
# machine, and `modal run --detach` only keeps the *currently running* remote
# function alive after the local client dies — so a locally-chained
# train → measure sequence breaks mid-pipeline. This function chains both
# phases remotely in a single detached invocation.

@app.function(
    image=image_train,
    gpu="A100-40GB",
    timeout=14400,     # 4h: train ~75 min + measure ~15 min, generous margin
    volumes={"/data085": vol_085},
    memory=32768,
    retries=10,        # spot preemption killed the 23:42 run at step ~170;
                       # train.py resumes from the latest step_* checkpoint
)
def train_and_measure():
    """Run train then measure in-process on one remote worker."""
    train_model.local()
    measure_model.local()
    print("train_and_measure DONE", flush=True)


# ─── collect results ──────────────────────────────────────────────────────────

@app.function(
    image=image_train,
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
    valid = {"generate", "train", "measure", "finish", "results", "all"}
    if phase not in valid:
        raise SystemExit(f"phase must be one of: {valid}. Got: {phase!r}")

    print(f"=== exp-085 Generational Transmission — phase={phase} ===")
    print(f"  Source: exp062-data / {CNAT_S0_CKPT}")
    print(f"  Corpus: exp085-data / {CORPUS_NAME}")
    print(f"  Run:    exp085-data / runs/{RUN_NAME}")
    print()

    if phase in ("generate", "all"):
        print("→ generate_corpus (torch.compile + StaticCache) ...", flush=True)
        generate_corpus.remote()
        print("  generate_corpus complete", flush=True)

    if phase in ("train", "all"):
        print("→ train_model ...", flush=True)
        train_model.remote()
        print("  train_model complete", flush=True)

    if phase == "finish":
        print("→ train_and_measure (remote-chained, detach-safe) ...", flush=True)
        train_and_measure.remote()
        print("  train_and_measure complete", flush=True)

    if phase in ("measure", "all"):
        print("→ measure_model ...", flush=True)
        measure_model.remote()
        print("  measure_model complete", flush=True)

    if phase in ("results", "all"):
        print("→ collect_results ...", flush=True)
        res = collect_results.remote()
        print("\n=== RESULTS ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        # Write summary to local results.json
        out = SCRIPT_DIR / "results.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nWrote {out}")

    print(f"\nexp-085 phase={phase} DONE.")
