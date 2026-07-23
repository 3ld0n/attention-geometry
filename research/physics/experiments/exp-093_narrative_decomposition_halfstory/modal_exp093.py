"""
exp-093 — Narrative Decomposition: Half-Story Ordering (Modal cloud launcher).

Pre-registration: notes.md (committed before this script ran).

Corpus: C-NAT-halfstory.bin (always-swap of story halves)
Seeds: 1400, 1401, 1402 (init seeds; data-seed 2400)

Phases:
  generate   : generate C-NAT-halfstory corpus on Modal (~15-20 min, CPU)
  seeds      : train + measure seeds 1400/1401/1402 in parallel
  control    : randomized-weights control on seed-1400 checkpoint
  results    : collect all measurement JSONs from volume and print summary
  all        : generate, then all 3 seeds (parallel), then control

Usage (from repo root):
    .venv/bin/python3 -m modal run --detach \\
        research/physics/experiments/exp-093_narrative_decomposition_halfstory/modal_exp093.py \\
        --phase all

    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-093_narrative_decomposition_halfstory/modal_exp093.py \\
        --phase results

Ariel — July 23, 2026. Pre-registered before first run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# ─── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"

# ─── Modal ─────────────────────────────────────────────────────────────────────
app = modal.App("exp093-narrative-halfstory")

vol = modal.Volume.from_name("exp093-halfstory-data", create_if_missing=True)

image_run = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
        "datasets==3.6.0",
    )
    .add_local_file(str(SCRIPT_DIR / "gen_cnat_halfstory.py"),
                    remote_path="/exp093/gen_cnat_halfstory.py")
    .add_local_file(str(EXP062_DIR / "train.py"),
                    remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"),
                    remote_path="/exp062/measure.py")
)

# ─── constants (pre-registered in notes.md) ────────────────────────────────────
CORPUS = "C-NAT-halfstory.bin"
SEEDS  = {1400: "run_halfstory_s0", 1401: "run_halfstory_s1", 1402: "run_halfstory_s2"}
DATA_SEED = 2400


# ─── helper: exec a patched script in a clean namespace ────────────────────────

def _run_patched(script_path: str, out_override: str, argv: list[str]) -> None:
    source = open(script_path).read()
    source = source.replace(
        "OUT = Path(__file__).resolve().parent",
        f"OUT = Path('{out_override}')",
    )
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": script_path}
    exec(compile(source, script_path, "exec"), ns)  # noqa: S102


# ─── generate phase ─────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu=None,
    timeout=7200,
    volumes={"/data093": vol},
    memory=8192,
    retries=3,
)
def generate_corpus():
    """Generate C-NAT-halfstory corpus on Modal."""
    import os
    output_path = f"/data093/{CORPUS}"
    if os.path.exists(output_path):
        print(f"generate SKIP — {output_path} already exists", flush=True)
        return
    _run_patched(
        "/exp093/gen_cnat_halfstory.py",
        out_override="/data093",
        argv=["gen_cnat_halfstory.py"],
    )
    vol.commit()
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"generate DONE. {output_path} ({size_gb:.2f} GB)", flush=True)


# ─── train + measure: single seed ─────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=14400,   # 4h: train (~75 min) + measure (~15 min) + margin
    volumes={"/data093": vol},
    memory=32768,
    retries=5,
)
def seed_run(init_seed: int):
    """Train + measure one seed on C-NAT-halfstory."""
    run_name    = SEEDS[init_seed]
    corpus_path = f"/data093/{CORPUS}"
    _run_patched(
        "/exp062/train.py",
        out_override="/data093",
        argv=[
            "train.py", corpus_path, run_name,
            f"--init-seed={init_seed}",
            f"--data-seed={DATA_SEED}",
            "--micro-batch=64",
            "--device=cuda",
        ],
    )
    vol.commit()
    _run_patched(
        "/exp062/measure.py",
        out_override="/data093",
        argv=[
            "measure.py",
            f"/data093/runs/{run_name}/step_2000",
            run_name,
            "--full-vocab",
        ],
    )
    vol.commit()
    print(f"seed_run {init_seed} ({run_name}) DONE", flush=True)


# ─── randomized-weights control (pre-registered) ──────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data093": vol},
    memory=32768,
)
def control():
    """Randomized-weights control on seed-1400 checkpoint."""
    import shutil
    import torch
    from transformers import GPTNeoXForCausalLM

    run_name  = SEEDS[1400]
    ckpt_path = f"/data093/runs/{run_name}/step_2000"
    rand_path = f"/data093/runs/{run_name}/step_2000_randomized"

    model = GPTNeoXForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(1400 + 1)
        for _, p in model.named_parameters():
            rnd = torch.randn(p.shape, generator=g, dtype=torch.float32)
            p.copy_(rnd * p.float().std())
    if Path(rand_path).exists():
        shutil.rmtree(rand_path)
    model.save_pretrained(rand_path)
    del model

    _run_patched(
        "/exp062/measure.py",
        out_override="/data093",
        argv=[
            "measure.py", rand_path,
            f"{run_name}_randcontrol",
            "--full-vocab",
        ],
    )
    vol.commit()
    print("control DONE", flush=True)


# ─── orchestrator ──────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu=None,
    timeout=21600,   # 6h ceiling for generate + 3 seeds + control
    volumes={"/data093": vol},
    memory=4096,
    retries=2,
)
def run_all():
    """Orchestrate generate → 3 seeds (parallel) → control, chained inside Modal."""
    generate_corpus.remote()
    # spawn all three seeds in parallel
    handles = [seed_run.spawn(s) for s in sorted(SEEDS)]
    for h in handles:
        h.get()
    # run control after seeds complete
    control.remote()
    print("run_all DONE", flush=True)


# ─── collect results ────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    timeout=120,
    volumes={"/data093": vol},
    memory=4096,
)
def collect_results() -> dict:
    """Read all measurement JSONs from volume and summarize."""
    out: dict = {}

    for seed, run_name in SEEDS.items():
        p = Path(f"/data093/measurements/{run_name}.json")
        if not p.exists():
            out[run_name] = "not found"
            continue
        d = json.loads(p.read_text())
        out[run_name] = {
            "init_seed": seed,
            "n_conformal": d.get("n_conformal", "?"),
            "n_syk_near":  d.get("n_syk_near",  "?"),
            "n_deep_L3_L5": d.get("n_deep_L3_L5", "?"),
            "n_backbone_L0": d.get("n_backbone_L0", "?"),
            "delta_median_conformal": d.get("delta_median_conformal", "?"),
            "layer_dist": d.get("layer_dist", {}),
        }

    # also check for control
    ctrl_run = SEEDS[1400]
    p_ctrl = Path(f"/data093/measurements/{ctrl_run}_randcontrol.json")
    if p_ctrl.exists():
        d = json.loads(p_ctrl.read_text())
        out[f"{ctrl_run}_randcontrol"] = {
            "n_conformal": d.get("n_conformal", "?"),
            "n_syk_near":  d.get("n_syk_near",  "?"),
        }

    return out


# ─── local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "all"):
    valid = {"generate", "seeds", "control", "results", "all"}
    if phase not in valid:
        print(f"phase must be one of: {valid}")
        raise SystemExit(1)

    if phase == "results":
        r = collect_results.remote()
        print(json.dumps(r, indent=2))

    elif phase == "generate":
        generate_corpus.remote()

    elif phase == "seeds":
        handles = [seed_run.spawn(s) for s in sorted(SEEDS)]
        print(f"Spawned seeds: {[h.object_id for h in handles]}")

    elif phase == "control":
        control.remote()
        print("control DONE")

    elif phase == "all":
        print("Phase: run_all (generate → 3 seeds → control, chained in Modal)")
        h = run_all.spawn()
        print(f"run_all handle: {h.object_id}")
        h.get()
        print("all DONE")
