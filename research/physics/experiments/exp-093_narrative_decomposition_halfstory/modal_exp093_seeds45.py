"""
exp-093 seed-robustness extension — seeds 1403 and 1404.

Pre-registered in notes.md (seed-robustness extension section, 2026-07-23 ~12:25 AM MDT)
before this script ran.

Uses the same corpus (C-NAT-halfstory.bin) and volume (exp093-halfstory-data) as the
original 3-seed run. Run names: run_halfstory_s3 (seed 1403) and run_halfstory_s4 (seed 1404).

Usage (from repo root):
    .venv/bin/python3 -m modal run --detach \\
        research/physics/experiments/exp-093_narrative_decomposition_halfstory/modal_exp093_seeds45.py \\
        --phase seeds

    .venv/bin/python3 -m modal run \\
        research/physics/experiments/exp-093_narrative_decomposition_halfstory/modal_exp093_seeds45.py \\
        --phase results
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
app = modal.App("exp093-halfstory-s45")

# Same volume as the original run
vol = modal.Volume.from_name("exp093-halfstory-data", create_if_missing=False)

image_run = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
        "datasets==3.6.0",
    )
    .add_local_file(str(EXP062_DIR / "train.py"),
                    remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"),
                    remote_path="/exp062/measure.py")
)

# ─── constants (pre-registered in notes.md) ────────────────────────────────────
CORPUS    = "C-NAT-halfstory.bin"
EXTRA_SEEDS = {1403: "run_halfstory_s3", 1404: "run_halfstory_s4"}
DATA_SEED = 2400  # same as original 3-seed run


def _run_patched(script_path: str, out_override: str, argv: list[str]) -> None:
    source = open(script_path).read()
    source = source.replace(
        "OUT = Path(__file__).resolve().parent",
        f"OUT = Path('{out_override}')",
    )
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": script_path}
    exec(compile(source, script_path, "exec"), ns)  # noqa: S102


# ─── train + measure: single seed ─────────────────────────────────────────────

@app.function(
    image=image_run,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data093": vol},
    memory=32768,
    retries=5,
)
def seed_run_extra(init_seed: int):
    """Train + measure one extra seed on C-NAT-halfstory."""
    run_name    = EXTRA_SEEDS[init_seed]
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
    print(f"seed_run_extra {init_seed} ({run_name}) DONE", flush=True)


# ─── collect all 5 results ────────────────────────────────────────────────────

@app.function(
    image=image_run,
    timeout=120,
    volumes={"/data093": vol},
    memory=4096,
)
def collect_all_results() -> dict:
    """Read measurement JSONs for all 5 seeds (1400-1404) and summarize."""
    all_seeds = {
        1400: "run_halfstory_s0",
        1401: "run_halfstory_s1",
        1402: "run_halfstory_s2",
        1403: "run_halfstory_s3",
        1404: "run_halfstory_s4",
    }
    out: dict = {}
    for seed, run_name in all_seeds.items():
        p = Path(f"/data093/measurements/{run_name}.json")
        if not p.exists():
            out[run_name] = "not found"
            continue
        d = json.loads(p.read_text())
        out[run_name] = {
            "init_seed": seed,
            "n_conformal": d.get("n_conformal", "?"),
            "n_syk_near": d.get("n_syk_near", "?"),
            "n_deep_L3_L5": d.get("n_deep_L3_L5", "?"),
            "n_backbone_L0": d.get("n_backbone_L0", "?"),
            "delta_median_conformal": d.get("delta_median_conformal", "?"),
            "layer_dist": d.get("layer_dist", {}),
        }
    return out


# ─── local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "seeds"):
    valid = {"seeds", "results"}
    if phase not in valid:
        print(f"phase must be one of: {valid}")
        raise SystemExit(1)

    if phase == "seeds":
        handles = [seed_run_extra.spawn(s) for s in sorted(EXTRA_SEEDS)]
        print(f"Spawned extra seeds {sorted(EXTRA_SEEDS)}: handles {[h.object_id for h in handles]}")
        print("Use --phase results to collect once both are done.")

    elif phase == "results":
        r = collect_all_results.remote()
        print(json.dumps(r, indent=2))
