"""
exp-089 — Huginn Latent-Iteration RG Flow (Modal cloud launcher).

Pre-registration of record: notes.md (committed 2026-07-20, c359b93a).

Protocol note (device deviation, recorded per pre-registration discipline):
The pre-registered protocol named MPS (Apple Silicon M5 Max) as the device.
The local ~16GB model download was bandwidth-bound (~200 KB/s measured on the
valley link, ≈21h ETA), so the run moved to Modal CUDA the same evening the
billing unblocked. Device is execution metadata: hypotheses, probe steps,
inputs, seeds, fitting protocol, and kill criteria are unchanged. The script
run remotely is the pre-registered run_latent_rg_flow.py with exactly two
mechanical patches applied at launch time (visible in _patched_source below):
  1. DEVICE_STR "mps"→"cuda"
  2. RESULTS_FILE → /results/results.json on the exp089-data volume

Usage (from repo root):
  .venv/bin/python -m modal run \\
    research/physics/experiments/exp-089_huginn_latent_rg_flow/modal_exp089.py

Ariel — July 20, 2026.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent

app = modal.App("exp089-huginn-latent-rg-flow")

vol_089 = modal.Volume.from_name("exp089-data", create_if_missing=True)
vol_hf = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        # Huginn's trust_remote_code modeling (raven_modeling_minimal.py) declares
        # _tied_weights_keys as a list; transformers v5 requires a dict there.
        # Pin to 4.x — the API generation Huginn was written against.
        "transformers==4.51.3",
        "datasets",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf-cache"})
    .add_local_file(
        str(SCRIPT_DIR / "run_latent_rg_flow.py"),
        remote_path="/exp089/run_latent_rg_flow.py",
    )
)


def _patched_source(source: str) -> str:
    """The two mechanical patches recorded in the protocol note above."""
    patched = source.replace(
        'DEVICE_STR = "mps" if torch.backends.mps.is_available() else "cpu"',
        'DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"',
    )
    patched = patched.replace(
        'RESULTS_FILE = SCRIPT_DIR / "results.json"',
        'RESULTS_FILE = Path("/results/results.json")',
    )
    assert patched != source, "patch targets not found — script changed?"
    return patched


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,
    volumes={"/results": vol_089, "/hf-cache": vol_hf},
    memory=32768,
    retries=2,
)
def run_experiment() -> dict:
    import sys

    source = open("/exp089/run_latent_rg_flow.py").read()
    source = _patched_source(source)

    sys.argv = ["run_latent_rg_flow.py"]
    ns: dict = {"__name__": "__main__", "__file__": "/exp089/run_latent_rg_flow.py"}
    exec(compile(source, "/exp089/run_latent_rg_flow.py", "exec"), ns)  # noqa: S102

    vol_089.commit()
    return json.loads(Path("/results/results.json").read_text())


@app.local_entrypoint()
def main():
    print("=== exp-089 Huginn Latent-Iteration RG Flow (Modal) ===")
    result = run_experiment.remote()

    out = SCRIPT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    print(f"Verdict: {result.get('verdict')}")
    for cond in ("nat_summary", "rand_summary"):
        s = result.get(cond, {})
        print(
            f"  {cond}: rho_convergence={s.get('rho_convergence')}, "
            f"rho_emergence={s.get('rho_emergence')}"
        )
