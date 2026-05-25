"""
exp-032 — Apple sigmoid-attention infrastructure probe.

Pre-stated question: Can we run the exp-007 per-head Δ protocol on Apple's
paired 7B sigmoid vs softmax checkpoints on this machine (M5 Max, 128GB)?

This script records what was tried and what blocked measurement — not a substitute
for the falsification experiment itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SIGMOID_REPO = "https://github.com/apple/ml-sigmoid-attention"
AXLEARN_INSTALL = (
    'pip install --ignore-installed "axlearn[core,apple-silicon] '
    '@ git+https://github.com/apple/axlearn.git"'
)
CHECKPOINTS = {
    "7b-sigmoid": {
        "gcs": (
            "gs://axlearn-public/experiments/"
            "gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/checkpoints/step_00250000"
        ),
        "config": "gala-sigmoid-7B-4k-hybridnorm-alibi-sp-rp",
        "pos_enc": "ALiBi (hybrid norm)",
        "attention": "sigmoid (unnormalized per Apple ICLR 2025)",
    },
    "7b-softmax": {
        "gcs": (
            "gs://axlearn-public/experiments/"
            "gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/checkpoints/step_00250000"
        ),
        "config": "gala-7B-hybridnorm-alibi-flash-sp-rp",
        "pos_enc": "ALiBi",
        "attention": "softmax",
    },
}


def run_cmd(cmd: list[str] | str, timeout: int = 120) -> dict:
    if isinstance(cmd, str):
        cmd = cmd.split()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": r.returncode,
            "stdout": (r.stdout or "")[-2000:],
            "stderr": (r.stderr or "")[-2000:],
        }
    except Exception as e:
        return {"cmd": " ".join(cmd), "error": str(e)}


def probe_openelm() -> dict:
    """Warm-up path from queue: apple/OpenELM-270M on HuggingFace."""
    record = {"model": "apple/OpenELM-270M", "purpose": "infrastructure warm-up (RoPE, not sigmoid)"}
    try:
        from transformers import AutoConfig

        try:
            AutoConfig.from_pretrained("apple/OpenELM-270M", trust_remote_code=True)
            record["config_load"] = "ok"
        except Exception as e:
            record["config_load"] = f"failed: {e}"
    except Exception as e:
        record["import_error"] = str(e)
    return record


def main() -> None:
    results = {
        "experiment": "exp-032",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": "Apple 7B sigmoid checkpoints are accessible locally for per-head Δ measurement.",
        "verdict": "aborted",
        "blockers": [],
        "attempts": [],
        "checkpoints_documented": CHECKPOINTS,
        "next_steps": [
            "Install axlearn per ml-sigmoid-attention notebook (may need Python 3.11 venv — tensorflow-io pin fails on 3.12).",
            "gcloud auth + download gs://axlearn-public/experiments/... checkpoints (~7B each).",
            "Extract attention with AXLearn InferenceRunner; adapt exp-007 decay protocol.",
            "Paired 7b-softmax baseline on same ALiBi stack for direct mechanism comparison.",
        ],
    }

    results["attempts"].append(
        {
            "step": "gsutil availability",
            **run_cmd(["which", "gsutil"]),
        }
    )
    if not shutil.which("gsutil"):
        results["blockers"].append("gsutil not installed — cannot fetch GCS checkpoints without gcloud SDK.")

    results["attempts"].append(
        {
            "step": "huggingface sigmoid search",
            "note": "No apple/* sigmoid weights on HuggingFace (only OpenELM RoPE family).",
        }
    )

    axlearn_pip = run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "axlearn[core,apple-silicon]",
            f"git+https://github.com/apple/axlearn.git",
        ],
        timeout=90,
    )
    results["attempts"].append({"step": "axlearn pip install (90s cap)", **axlearn_pip})
    if axlearn_pip.get("returncode", 1) != 0:
        results["blockers"].append(
            "axlearn[core,apple-silicon] pip install failed (tensorflow-io==0.37.3 unavailable on Python 3.12)."
        )

    results["openelm_probe"] = probe_openelm()
    if "failed" in str(results["openelm_probe"].get("config_load", "")):
        results["blockers"].append(
            "OpenELM-270M: transformers 5.8.1 + remote code — OpenELMConfig rejects use_cache in __post_init__."
        )

    results["reference_protocol"] = "exp-007 / exp-031 (GPT-2 per-head, R²>0.90, median Δ)"
    results["reference_softmax"] = {"gpt2": 0.2493, "gpt2-medium": 0.2589, "gpt2-large_global": 0.2819}

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("verdict", "blockers", "next_steps")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
