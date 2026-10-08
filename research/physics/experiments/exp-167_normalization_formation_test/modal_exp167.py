"""
exp-167 — Modal cloud launcher: normalization formation test.

Trains three normalization conditions (ln / none / rms) on C-NAT_full.bin
from the exp062-data volume, using A100 GPU. Runs the frozen-protocol census
at step 2000 for each condition and downloads results.

Pre-registration: prereg.md (committed 61431c9 before this script existed).

Volumes:
  exp062-data — source of corpora/C-NAT_full.bin (read-only)
  exp167-data — training runs and results (read-write, created if missing)

Usage (from repo root):
    .venv/bin/python3 -m modal run research/physics/experiments/\\
exp-167_normalization_formation_test/modal_exp167.py --phase all

    .venv/bin/python3 -m modal run --detach ... --phase all

Phases:
  train_a   — train Condition A (LN, standard lr)
  train_b   — train Condition B (NoNorm, lower lr)
  train_c   — train Condition C (RMSNorm, standard lr)
  measure_a — census at step 2000 for Condition A
  measure_b — census at step 2000 for Condition B (or last stable ckpt)
  measure_c — census at step 2000 for Condition C
  all       — train_a → measure_a → train_b → measure_b → train_c → measure_c
  results   — print results from the volume
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# ─── infrastructure ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

app = modal.App("exp167-normalization-formation")

vol_062 = modal.Volume.from_name("exp062-data")          # source corpus
vol_167 = modal.Volume.from_name("exp167-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
        "safetensors>=0.4",
    )
    .add_local_file(str(SCRIPT_DIR / "train_norm.py"),   remote_path="/exp167/train_norm.py")
    .add_local_file(str(SCRIPT_DIR / "measure_norm.py"), remote_path="/exp167/measure_norm.py")
)

CORPUS_PATH = "/vol062/corpora/C-NAT_full.bin"
RUNS_ROOT   = "/vol167/runs"
RESULTS_DIR = "/vol167/results"


# ─── training functions ────────────────────────────────────────────────────────

def _train(norm: str, gpu: str = "A100") -> dict:
    """Train one normalization condition on the A100."""
    import subprocess, json
    from pathlib import Path

    run_name = f"exp167_{norm}"
    out_dir = Path(RUNS_ROOT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write norm_condition.txt so measure script can auto-detect
    (out_dir / "norm_condition.txt").write_text(norm + "\n")

    cmd = [
        sys.executable, "/exp167/train_norm.py",
        CORPUS_PATH,
        "--norm", norm,
        "--run-name", run_name,
        "--runs-root", RUNS_ROOT,
        "--micro-batch", "64",
        "--device", "cuda",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"train_norm.py failed for norm={norm}")
    vol_167.commit()
    # Parse training summary
    summary_path = out_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "unknown"}
    return summary


def _measure(norm: str) -> dict:
    """Run the frozen-protocol census for one condition."""
    import subprocess, json
    from pathlib import Path

    run_name = f"exp167_{norm}"
    ckpt_dir = Path(RUNS_ROOT) / run_name

    # Find the step_2000 checkpoint (or last stable if B diverged)
    ckpts = sorted(ckpt_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    # Prefer step_2000; fall back to last checkpoint
    target = next((c for c in ckpts if c.name == "step_2000"), ckpts[-1])
    print(f"Measuring norm={norm} at checkpoint {target.name}")

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_tag = f"{norm}_{target.name}"

    cmd = [
        sys.executable, "/exp167/measure_norm.py",
        str(target),
        "--norm", norm,
        "--out-tag", out_tag,
        "--results-dir", RESULTS_DIR,
        "--device", "cuda",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"measure_norm.py failed for norm={norm}")
    vol_167.commit()
    result_path = Path(RESULTS_DIR) / f"{out_tag}.json"
    return json.loads(result_path.read_text()) if result_path.exists() else {}


@app.function(
    image=image,
    volumes={"/vol062": vol_062, "/vol167": vol_167},
    gpu="A100",
    timeout=7200,
)
def train_condition(norm: str) -> dict:
    return _train(norm)


@app.function(
    image=image,
    volumes={"/vol062": vol_062, "/vol167": vol_167},
    gpu="A100",
    timeout=3600,
)
def measure_condition(norm: str) -> dict:
    return _measure(norm)


@app.function(
    image=image,
    volumes={"/vol062": vol_062, "/vol167": vol_167},
    gpu="A100",
    timeout=21600,  # 6 hours for all three conditions
)
def run_all() -> dict:
    """Train and measure all three conditions sequentially."""
    summaries = {}
    for norm in ["ln", "none", "rms"]:
        print(f"\n{'='*60}")
        print(f"Training Condition {norm.upper()} ...")
        print(f"{'='*60}")
        train_summary = _train(norm)
        summaries[f"train_{norm}"] = train_summary

        print(f"\nMeasuring Condition {norm.upper()} ...")
        measure_result = _measure(norm)
        summaries[f"measure_{norm}"] = {
            "n_conformal": measure_result.get("n_conformal"),
            "n_syk_near": measure_result.get("n_syk_near"),
            "delta_median_all": measure_result.get("delta_median_all"),
        }
        print(f"  n_conformal={measure_result.get('n_conformal')} "
              f"n_syk_near={measure_result.get('n_syk_near')}")

    # Write combined summary
    from pathlib import Path
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (Path(RESULTS_DIR) / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    vol_167.commit()
    return summaries


# ─── local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(phase: str = "all", norm: str = "ln") -> None:
    """
    phase: train_a | train_b | train_c | measure_a | measure_b | measure_c | all | results
    norm:  ln | none | rms  (used only for train_X / measure_X phases)
    """
    if phase == "all":
        result = run_all.remote()
        print("\n=== exp-167 combined summary ===")
        for cond in ["ln", "none", "rms"]:
            m = result.get(f"measure_{cond}", {})
            t = result.get(f"train_{cond}", {})
            status = t.get("status", "?")
            print(f"  {cond:4s}: n_conformal={m.get('n_conformal','?'):3}  "
                  f"n_syk_near={m.get('n_syk_near','?'):3}  "
                  f"Δ_med={m.get('delta_median_all','?')}  "
                  f"train_status={status}")

    elif phase.startswith("train_"):
        cond = {"train_a": "ln", "train_b": "none", "train_c": "rms"}.get(phase, norm)
        result = train_condition.remote(cond)
        print(f"\nTraining summary for {cond}:", result)

    elif phase.startswith("measure_"):
        cond = {"measure_a": "ln", "measure_b": "none", "measure_c": "rms"}.get(phase, norm)
        result = measure_condition.remote(cond)
        print(f"\nCensus results for {cond}:")
        print(f"  n_conformal  : {result.get('n_conformal')} / {result.get('n_total_heads')}")
        print(f"  n_syk_near   : {result.get('n_syk_near')} / {result.get('n_total_heads')}")
        print(f"  delta_median : {result.get('delta_median_all')}")

    elif phase == "results":
        # Download and print existing results from the volume
        import subprocess
        r = subprocess.run(
            ["python3", "-m", "modal", "volume", "ls", "exp167-data", "results/"],
            capture_output=True, text=True
        )
        print(r.stdout or "(no results yet)")

    else:
        print(f"Unknown phase: {phase!r}")
        print("Valid phases: all | train_a | train_b | train_c | "
              "measure_a | measure_b | measure_c | results")
