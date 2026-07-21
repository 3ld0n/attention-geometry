"""
exp-085 multi-seed extension — train runs s1/s2 on the existing C-generated corpus.

Declared in notes.md ("Multi-seed extension", 2026-07-21) BEFORE launch.
Same corpus, same protocol as run_Cgen_s0; only training seeds vary,
matching exp-062's C-NAT convention (init 1001/1002, data 2001/2002).

Usage (detached so the runs survive this machine):
    .venv/bin/python3 -m modal run --detach research/physics/experiments/exp-085_generational_transmission/modal_exp085_multiseed.py --run s1
    .venv/bin/python3 -m modal run --detach research/physics/experiments/exp-085_generational_transmission/modal_exp085_multiseed.py --run s2

Collect when done:
    .venv/bin/python3 -m modal run research/physics/experiments/exp-085_generational_transmission/modal_exp085_multiseed.py --run s1 --collect-only true

Ariel — July 21, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
EXP062_DIR = SCRIPT_DIR.parent / "exp-062_corpus_statistics"

app = modal.App("exp085-multiseed")
vol_085 = modal.Volume.from_name("exp085-data", create_if_missing=False)

image_train = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "torch==2.12.0",
        "transformers==5.8.1",
    )
    .add_local_file(str(EXP062_DIR / "train.py"), remote_path="/exp062/train.py")
    .add_local_file(str(EXP062_DIR / "measure.py"), remote_path="/exp062/measure.py")
)

CORPUS_NAME = "C-generated_s0.bin"
SEEDS = {
    "s1": {"run_name": "run_Cgen_s1", "init_seed": 1001, "data_seed": 2001},
    "s2": {"run_name": "run_Cgen_s2", "init_seed": 1002, "data_seed": 2002},
}


def _run_patched(script_path: str, out_override: str, argv: list[str]) -> None:
    """Identical to modal_exp085.py: exec the frozen script with OUT redirected."""
    source = open(script_path).read()
    source = source.replace(
        "OUT = Path(__file__).resolve().parent",
        f"OUT = Path('{out_override}')",
    )
    sys.argv = argv
    ns: dict = {"__name__": "__main__", "__file__": script_path}
    exec(compile(source, script_path, "exec"), ns)  # noqa: S102


@app.function(
    image=image_train,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data085": vol_085},
    memory=32768,
    retries=10,  # spot preemption; train.py resumes from latest step_* checkpoint
)
def train_and_measure(run_name: str, init_seed: int, data_seed: int):
    corpus_path = f"/data085/{CORPUS_NAME}"
    _run_patched(
        "/exp062/train.py",
        out_override="/data085",
        argv=[
            "train.py",
            corpus_path,
            run_name,
            f"--init-seed={init_seed}",
            f"--data-seed={data_seed}",
            "--micro-batch=64",
            "--device=cuda",
        ],
    )
    vol_085.commit()
    print(f"train DONE ({run_name})", flush=True)

    ckpt_path = f"/data085/runs/{run_name}/step_2000"
    _run_patched(
        "/exp062/measure.py",
        out_override="/data085",
        argv=["measure.py", ckpt_path, run_name, "--full-vocab"],
    )
    vol_085.commit()
    print(f"measure DONE ({run_name})", flush=True)


@app.function(image=image_train, timeout=120, volumes={"/data085": vol_085}, memory=4096)
def collect(run_name: str) -> dict:
    result_path = Path(f"/data085/measurements/{run_name}.json")
    if not result_path.exists():
        return {"error": f"{result_path} not found"}
    d = json.loads(result_path.read_text())
    return {
        "run": run_name,
        "n_conformal": d["n_conformal"],
        "n_syk_near": d["n_syk_near"],
        "delta_median_conformal": d["delta_median_conformal"],
        "forms": d["forms"],
    }


@app.local_entrypoint()
def main(run: str = "s1", collect_only: bool = False):
    if run not in SEEDS:
        raise SystemExit(f"--run must be one of {sorted(SEEDS)}")
    cfg = SEEDS[run]
    if collect_only:
        print(json.dumps(collect.remote(cfg["run_name"]), indent=1))
        return
    print(f"=== exp-085 multi-seed: {cfg['run_name']} "
          f"(init {cfg['init_seed']}, data {cfg['data_seed']}) ===")
    train_and_measure.remote(cfg["run_name"], cfg["init_seed"], cfg["data_seed"])
    print(json.dumps(collect.remote(cfg["run_name"]), indent=1))
