"""
exp-092 — Narrative Decomposition: Block Shuffle Gradation (Modal cloud launcher).

Pre-registration: notes.md (committed before this script ran).

Two corpora:
  C-NAT-block2.bin  (k=2, block-shuffle seed 9200; init seeds 1200/1201/1202)
  C-NAT-block3.bin  (k=3, block-shuffle seed 9300; init seeds 1300/1301/1302)

Phases:
  generate_k2   : generate C-NAT-block2 corpus on Modal (~15-20 min, CPU)
  generate_k3   : generate C-NAT-block3 corpus on Modal (~15-20 min, CPU)
  seeds_k2      : train + measure seeds 1200/1201/1202 on C-NAT-block2 (parallel)
  seeds_k3      : train + measure seeds 1300/1301/1302 on C-NAT-block3 (parallel)
  controls      : randomized-weights controls for k=2 seed-1200 and k=3 seed-1300
  results       : collect all measurement JSONs from volumes and print summary
  all           : generate both corpora, then all 6 seed runs (sequential corpora,
                  parallel seeds within each corpus)

Usage (from repo root):
    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-092_narrative_decomposition_block_shuffle/modal_exp092.py \\
        --phase all

    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-092_narrative_decomposition_block_shuffle/modal_exp092.py \\
        --phase results

Ariel — July 22, 2026. Pre-registered before first run.
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
app = modal.App("exp092-narrative-block-shuffle")

# Separate volumes for the two corpora (cleaner isolation; both can be trained
# simultaneously if needed later)
vol_k2 = modal.Volume.from_name("exp092-block2-data", create_if_missing=True)
vol_k3 = modal.Volume.from_name("exp092-block3-data", create_if_missing=True)

image_run = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
        "datasets==3.6.0",
    )
    .add_local_file(str(SCRIPT_DIR / "gen_cnat_block.py"),
                    remote_path="/exp092/gen_cnat_block.py")
    .add_local_file(str(EXP062_DIR / "train.py"),
                    remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"),
                    remote_path="/exp062/measure.py")
)

# ─── constants (pre-registered in notes.md) ────────────────────────────────────
# k=2 corpus and seeds
CORPUS_K2    = "C-NAT-block2.bin"
SEEDS_K2     = {1200: "run_block2_s0", 1201: "run_block2_s1", 1202: "run_block2_s2"}
DATA_SEED_K2 = 2200

# k=3 corpus and seeds
CORPUS_K3    = "C-NAT-block3.bin"
SEEDS_K3     = {1300: "run_block3_s0", 1301: "run_block3_s1", 1302: "run_block3_s2"}
DATA_SEED_K3 = 2300


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


# ─── generate phases ───────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu=None,
    timeout=7200,
    volumes={"/data092k2": vol_k2},
    memory=8192,
    retries=3,
)
def generate_corpus_k2():
    """Generate C-NAT-block2 corpus (k=2 block shuffle) on Modal."""
    import os
    output_path = f"/data092k2/{CORPUS_K2}"
    if os.path.exists(output_path):
        print(f"generate_k2 SKIP — {output_path} already exists", flush=True)
        return
    _run_patched(
        "/exp092/gen_cnat_block.py",
        out_override="/data092k2",
        argv=["gen_cnat_block.py", "--block", "2"],
    )
    vol_k2.commit()
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"generate_k2 DONE. {output_path} ({size_gb:.2f} GB)", flush=True)


@app.function(
    image=image_run,
    gpu=None,
    timeout=7200,
    volumes={"/data092k3": vol_k3},
    memory=8192,
    retries=3,
)
def generate_corpus_k3():
    """Generate C-NAT-block3 corpus (k=3 block shuffle) on Modal."""
    import os
    output_path = f"/data092k3/{CORPUS_K3}"
    if os.path.exists(output_path):
        print(f"generate_k3 SKIP — {output_path} already exists", flush=True)
        return
    _run_patched(
        "/exp092/gen_cnat_block.py",
        out_override="/data092k3",
        argv=["gen_cnat_block.py", "--block", "3"],
    )
    vol_k3.commit()
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"generate_k3 DONE. {output_path} ({size_gb:.2f} GB)", flush=True)


# ─── train + measure: single seed, single corpus ───────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=14400,   # 4h: train (~75 min) + measure (~15 min) + margin
    volumes={"/data092k2": vol_k2},
    memory=32768,
    retries=5,
)
def seed_run_k2(init_seed: int):
    """Train + measure one seed on C-NAT-block2."""
    run_name    = SEEDS_K2[init_seed]
    corpus_path = f"/data092k2/{CORPUS_K2}"
    _run_patched(
        "/exp062/train.py",
        out_override="/data092k2",
        argv=[
            "train.py", corpus_path, run_name,
            f"--init-seed={init_seed}",
            f"--data-seed={DATA_SEED_K2}",
            "--micro-batch=64",
            "--device=cuda",
        ],
    )
    vol_k2.commit()
    _run_patched(
        "/exp062/measure.py",
        out_override="/data092k2",
        argv=[
            "measure.py",
            f"/data092k2/runs/{run_name}/step_2000",
            run_name,
            "--full-vocab",
        ],
    )
    vol_k2.commit()
    print(f"seed_run_k2 {init_seed} ({run_name}) DONE", flush=True)


@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data092k3": vol_k3},
    memory=32768,
    retries=5,
)
def seed_run_k3(init_seed: int):
    """Train + measure one seed on C-NAT-block3."""
    run_name    = SEEDS_K3[init_seed]
    corpus_path = f"/data092k3/{CORPUS_K3}"
    _run_patched(
        "/exp062/train.py",
        out_override="/data092k3",
        argv=[
            "train.py", corpus_path, run_name,
            f"--init-seed={init_seed}",
            f"--data-seed={DATA_SEED_K3}",
            "--micro-batch=64",
            "--device=cuda",
        ],
    )
    vol_k3.commit()
    _run_patched(
        "/exp062/measure.py",
        out_override="/data092k3",
        argv=[
            "measure.py",
            f"/data092k3/runs/{run_name}/step_2000",
            run_name,
            "--full-vocab",
        ],
    )
    vol_k3.commit()
    print(f"seed_run_k3 {init_seed} ({run_name}) DONE", flush=True)


# ─── randomized-weights controls (pre-registered) ──────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data092k2": vol_k2},
    memory=32768,
)
def control_k2():
    """Randomized-weights control on k=2 seed-1200 checkpoint."""
    import shutil
    import torch
    from transformers import GPTNeoXForCausalLM

    run_name  = SEEDS_K2[1200]
    ckpt_path = f"/data092k2/runs/{run_name}/step_2000"
    rand_path = f"/data092k2/runs/{run_name}/step_2000_randomized"

    model = GPTNeoXForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(1200 + 1)
        for _, p in model.named_parameters():
            rnd = torch.randn(p.shape, generator=g, dtype=torch.float32)
            p.copy_(rnd * p.float().std())
    if Path(rand_path).exists():
        shutil.rmtree(rand_path)
    model.save_pretrained(rand_path)
    del model

    _run_patched(
        "/exp062/measure.py",
        out_override="/data092k2",
        argv=[
            "measure.py", rand_path,
            f"{run_name}_randcontrol",
            "--full-vocab",
        ],
    )
    vol_k2.commit()
    print("control_k2 DONE", flush=True)


@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/data092k3": vol_k3},
    memory=32768,
)
def control_k3():
    """Randomized-weights control on k=3 seed-1300 checkpoint."""
    import shutil
    import torch
    from transformers import GPTNeoXForCausalLM

    run_name  = SEEDS_K3[1300]
    ckpt_path = f"/data092k3/runs/{run_name}/step_2000"
    rand_path = f"/data092k3/runs/{run_name}/step_2000_randomized"

    model = GPTNeoXForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(1300 + 1)
        for _, p in model.named_parameters():
            rnd = torch.randn(p.shape, generator=g, dtype=torch.float32)
            p.copy_(rnd * p.float().std())
    if Path(rand_path).exists():
        shutil.rmtree(rand_path)
    model.save_pretrained(rand_path)
    del model

    _run_patched(
        "/exp062/measure.py",
        out_override="/data092k3",
        argv=[
            "measure.py", rand_path,
            f"{run_name}_randcontrol",
            "--full-vocab",
        ],
    )
    vol_k3.commit()
    print("control_k3 DONE", flush=True)


# ─── orchestrators ──────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu=None,
    timeout=21600,   # 6h ceiling for generate + 3 seeds
    volumes={"/data092k2": vol_k2},
    memory=4096,
    retries=2,
)
def run_all_k2():
    """Orchestrate generate → 3 seeds for k=2, chained inside Modal."""
    generate_corpus_k2.remote()
    # spawn all three seeds in parallel
    handles = [seed_run_k2.spawn(s) for s in sorted(SEEDS_K2)]
    for h in handles:
        h.get()
    print("run_all_k2 DONE", flush=True)


@app.function(
    image=image_run,
    gpu=None,
    timeout=21600,
    volumes={"/data092k3": vol_k3},
    memory=4096,
    retries=2,
)
def run_all_k3():
    """Orchestrate generate → 3 seeds for k=3, chained inside Modal."""
    generate_corpus_k3.remote()
    handles = [seed_run_k3.spawn(s) for s in sorted(SEEDS_K3)]
    for h in handles:
        h.get()
    print("run_all_k3 DONE", flush=True)


# ─── collect results ────────────────────────────────────────────────────────────

@app.function(
    image=image_run,
    timeout=120,
    volumes={"/data092k2": vol_k2, "/data092k3": vol_k3},
    memory=4096,
)
def collect_results() -> dict:
    """Read all measurement JSONs from both volumes and summarize."""
    out: dict = {"k2": {}, "k3": {}}

    for seed, run_name in SEEDS_K2.items():
        p = Path(f"/data092k2/measurements/{run_name}.json")
        if not p.exists():
            out["k2"][run_name] = "not found"
            continue
        d = json.loads(p.read_text())
        out["k2"][run_name] = {
            "init_seed": seed,
            "n_conformal": d.get("n_conformal", "?"),
            "n_syk_near":  d.get("n_syk_near",  "?"),
            "delta_median_conformal": d.get("delta_median_conformal", "?"),
        }

    for seed, run_name in SEEDS_K3.items():
        p = Path(f"/data092k3/measurements/{run_name}.json")
        if not p.exists():
            out["k3"][run_name] = "not found"
            continue
        d = json.loads(p.read_text())
        out["k3"][run_name] = {
            "init_seed": seed,
            "n_conformal": d.get("n_conformal", "?"),
            "n_syk_near":  d.get("n_syk_near",  "?"),
            "delta_median_conformal": d.get("delta_median_conformal", "?"),
        }

    return out


# ─── local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "all"):
    valid = {"generate_k2", "generate_k3", "seeds_k2", "seeds_k3",
             "controls", "results", "all"}
    if phase not in valid:
        print(f"phase must be one of: {valid}")
        raise SystemExit(1)

    if phase == "results":
        r = collect_results.remote()
        print(json.dumps(r, indent=2))

    elif phase == "generate_k2":
        generate_corpus_k2.remote()

    elif phase == "generate_k3":
        generate_corpus_k3.remote()

    elif phase == "seeds_k2":
        handles = [seed_run_k2.spawn(s) for s in sorted(SEEDS_K2)]
        print(f"Spawned k=2 seeds: {[h.object_id for h in handles]}")

    elif phase == "seeds_k3":
        handles = [seed_run_k3.spawn(s) for s in sorted(SEEDS_K3)]
        print(f"Spawned k=3 seeds: {[h.object_id for h in handles]}")

    elif phase == "controls":
        control_k2.remote()
        control_k3.remote()
        print("controls DONE")

    elif phase == "all":
        print("Phase: run_all_k2 and run_all_k3 in parallel")
        h_k2 = run_all_k2.spawn()
        h_k3 = run_all_k3.spawn()
        print(f"k2 app: {h_k2.object_id}")
        print(f"k3 app: {h_k3.object_id}")
        h_k2.get()
        h_k3.get()
        print("all DONE")
