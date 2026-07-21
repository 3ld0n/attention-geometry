"""
exp-090 — Ouro Latent-Iteration RG Flow (Modal cloud launcher).

Pre-registration of record: notes.md (committed 2026-07-21, 05293e6c, before any
run code existed). Unlike exp-089 the measurement script is CUDA-native, so no
source patching is needed — the script ships to the container verbatim.

Usage (from repo root):
  .venv/bin/python -m modal run \\
    research/physics/experiments/exp-090_ouro_rg_flow/modal_exp090.py
  .venv/bin/python -m modal run \\
    research/physics/experiments/exp-090_ouro_rg_flow/modal_exp090.py --control

Ariel — July 21, 2026.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent

app = modal.App("exp090-ouro-rg-flow")

vol_090 = modal.Volume.from_name("exp090-data", create_if_missing=True)
vol_hf = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        # Ouro's trust_remote_code modeling imports transformers.masking_utils,
        # modeling_layers.GenericFor*, check_model_inputs — the 4.55+ API
        # generation (config says transformers_version 4.55.0).
        "transformers==4.57.1",
        "datasets",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
    .add_local_file(
        str(SCRIPT_DIR / "run_ouro_rg_flow.py"),
        remote_path="/exp090/run_ouro_rg_flow.py",
    )
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,
    volumes={"/results": vol_090, "/hf-cache": vol_hf},
    memory=32768,
    retries=2,
)
def run_experiment(control: bool = False) -> dict:
    import subprocess
    import sys

    fname = "results_control.json" if control else "results.json"
    cmd = [sys.executable, "/exp090/run_ouro_rg_flow.py", "--out", f"/results/{fname}"]
    if control:
        cmd.append("--control")
    subprocess.run(cmd, check=True)

    vol_090.commit()
    return json.loads(Path(f"/results/{fname}").read_text())


@app.local_entrypoint()
def main(control: bool = False):
    label = " — randomized-weights control" if control else ""
    print(f"=== exp-090 Ouro Latent-Iteration RG Flow (Modal){label} ===")
    result = run_experiment.remote(control=control)

    fname = "results_control.json" if control else "results.json"
    out = SCRIPT_DIR / fname
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    print(f"Verdict: {result.get('verdict', '?')}")
    for cond in ("nat_summary", "rand_summary"):
        s = result.get(cond, {})
        if s:
            print(
                f"  {cond}: rho_convergence={s.get('rho_convergence')}, "
                f"rho_emergence={s.get('rho_emergence')}"
            )
