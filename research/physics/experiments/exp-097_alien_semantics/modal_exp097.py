"""
exp-097 — Alien Semantics: world-holding without real-world grounding (Modal cloud launcher).

Pre-registration: notes.md (committed before this script ran).

Corpus: C-alien.bin (procedural world-simulator stories; entities Flurps/Blurns/Zarbs)
Seeds: 1700, 1701, 1702 (init seeds; data-seed 2700)

Phases:
  generate   : generate C-alien corpus on Modal (~2-4h, CPU)
  seeds      : train + measure seeds 1700/1701/1702 in parallel
  control    : randomized-weights control on seed-1700 checkpoint
  results    : collect all measurement JSONs from volume and print summary
  all        : generate, then all 3 seeds (parallel), then control

Usage (from repo root, ALWAYS use --detach for long runs):
    .venv/bin/python3 -m modal run --detach \\
        research/physics/experiments/exp-097_alien_semantics/modal_exp097.py \\
        --phase all

    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-097_alien_semantics/modal_exp097.py \\
        --phase results

Ariel — July 27, 2026. Pre-registered before first run.
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
app = modal.App("exp097-alien-semantics")

vol = modal.Volume.from_name("exp097-alien-data", create_if_missing=True)

image_run = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
        "datasets==3.6.0",
    )
    .add_local_file(str(SCRIPT_DIR / "gen_calien.py"),
                    remote_path="/exp097/gen_calien.py")
    .add_local_file(str(EXP062_DIR / "train.py"),
                    remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"),
                    remote_path="/exp062/measure.py")
)

# ─── constants (pre-registered in notes.md) ────────────────────────────────────
CORPUS    = "C-alien.bin"
SEEDS     = {1700: "run_alien_s0", 1701: "run_alien_s1", 1702: "run_alien_s2"}
DATA_SEED = 2700


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
    timeout=14400,   # 4h: C-alien generation is pure Python, no GPU
    volumes={"/data097": vol},
    memory=8192,
    retries=3,
)
def generate_corpus():
    """Generate C-alien corpus on Modal (CPU, ~2-4h)."""
    import os
    output_path = f"/data097/{CORPUS}"
    if os.path.exists(output_path):
        print(f"generate SKIP — {output_path} already exists", flush=True)
        return
    _run_patched(
        "/exp097/gen_calien.py",
        out_override="/data097",
        argv=["gen_calien.py"],
    )
    vol.commit()
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"generate DONE. {output_path} ({size_gb:.2f} GB)", flush=True)


# ─── train + measure: single seed ─────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=14400,   # 4h: train (~75 min) + measure (~15 min) + margin
    volumes={"/data097": vol},
    memory=32768,
    retries=5,
)
def seed_run(init_seed: int):
    """Train + measure one seed on C-alien."""
    run_name    = SEEDS[init_seed]
    corpus_path = f"/data097/{CORPUS}"
    _run_patched(
        "/exp062/train.py",
        out_override="/data097",
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
        out_override="/data097",
        argv=[
            "measure.py",
            f"/data097/runs/{run_name}/step_2000",
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
    volumes={"/data097": vol},
    memory=32768,
)
def control():
    """Randomized-weights control on seed-1700 checkpoint."""
    import shutil
    import torch
    from transformers import GPTNeoXForCausalLM

    run_name  = SEEDS[1700]
    ckpt_path = f"/data097/runs/{run_name}/step_2000"
    rand_path = f"/data097/runs/{run_name}/step_2000_randomized"

    model = GPTNeoXForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(1700 + 1)
        for _, p in model.named_parameters():
            rnd = torch.randn(p.shape, generator=g, dtype=torch.float32)
            p.copy_(rnd * p.float().std())
    if Path(rand_path).exists():
        shutil.rmtree(rand_path)
    model.save_pretrained(rand_path)
    del model

    _run_patched(
        "/exp062/measure.py",
        out_override="/data097",
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
    timeout=72000,   # 20h ceiling for generate + 3 seeds + control
    volumes={"/data097": vol},
    memory=4096,
    retries=2,
)
def run_all():
    """Orchestrate generate → 3 seeds (parallel) → control, chained inside Modal."""
    generate_corpus.remote()
    handles = [seed_run.spawn(s) for s in sorted(SEEDS)]
    for h in handles:
        h.get()
    control.remote()
    print("run_all DONE", flush=True)


# ─── collect results ────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    timeout=120,
    volumes={"/data097": vol},
    memory=4096,
)
def collect_results() -> dict:
    """Read all measurement JSONs from volume and summarize."""
    out: dict = {}

    for seed, run_name in SEEDS.items():
        p = Path(f"/data097/measurements/{run_name}.json")
        if not p.exists():
            out[run_name] = "not found"
            continue
        d = json.loads(p.read_text())
        out[run_name] = {
            "init_seed": seed,
            "n_conformal":            d.get("n_conformal", "?"),
            "n_syk_near":             d.get("n_syk_near",  "?"),
            "n_deep_L3_L5":           d.get("n_deep_L3_L5", "?"),
            "n_backbone_L0":          d.get("n_backbone_L0", "?"),
            "delta_median_conformal": d.get("delta_median_conformal", "?"),
            "layer_dist":             d.get("layer_dist", {}),
        }

    # check for control
    ctrl_run = SEEDS[1700]
    p_ctrl = Path(f"/data097/measurements/{ctrl_run}_randcontrol.json")
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
