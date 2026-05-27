"""
exp-033 — Apple sigmoid falsification: infrastructure verification (May 26, 2026).

Pre-stated: paired 7B sigmoid vs softmax (ALiBi), exp-007 per-head Δ protocol.
Sigmoid Δ ≈ 0.25 → mechanism-independent universality; Δ ≠ 0.25 → softmax load-bearing.

This script records what unblocked since exp-032 and what remains before measurement.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "results.json"

SIGMOID_GCS = (
    "gs://axlearn-public/experiments/"
    "gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/checkpoints/step_00250000"
)
SOFTMAX_GCS = (
    "gs://axlearn-public/experiments/"
    "gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/checkpoints/step_00250000"
)


def run(cmd: list[str], timeout: int = 60) -> dict:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": " ".join(cmd),
            "returncode": r.returncode,
            "stdout": (r.stdout or "")[-1500:],
            "stderr": (r.stderr or "")[-1500:],
        }
    except Exception as e:
        return {"cmd": " ".join(cmd), "error": str(e)}


def main() -> None:
    record = {
        "experiment": "exp-033",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypothesis": "Infrastructure for Apple 7B sigmoid/softmax Δ measurement is ready.",
        "verdict": "partial",
        "corrections_from_exp_032": [],
        "remaining_blockers": [],
        "checkpoint_sizes_bytes": {
            "sigmoid_full_checkpoint": 72696058910,
            "sigmoid_model_only_gda": 24456148609,
            "note": "Notebook downloads tf_* + gda/model + index only (~model weights, not learner)",
        },
        "reference_delta": {"gpt2": 0.2493, "syk_prediction": 0.25},
        "attempts": [],
    }

    gsutil = shutil.which("gsutil")
    record["attempts"].append({"step": "gsutil", "path": gsutil})
    if gsutil:
        record["corrections_from_exp_032"].append(
            "gsutil IS installed (exp-032 reported missing — PATH or install changed since May 25)"
        )
        ls = run([gsutil, "ls", "gs://axlearn-public/experiments/"], timeout=30)
        record["attempts"].append({"step": "gcs_list", **ls})
        if ls.get("returncode") == 0:
            record["gcs_access"] = True
        else:
            record["remaining_blockers"].append("gcs_list failed — auth?")
    else:
        record["remaining_blockers"].append("gsutil not found")

    # axlearn import + trainer configs
    try:
        from axlearn.common.attention import SigmoidAttention  # noqa: F401
        from axlearn.common.config import get_named_trainer_config

        get_named_trainer_config(
            "gala-sigmoid-7B-4k-hybridnorm-alibi-sp-rp",
            config_module="axlearn.experiments.text.gpt.pajama_sigmoid_trainer",
        )()
        get_named_trainer_config(
            "gala-7B-hybridnorm-alibi-flash-sp-rp",
            config_module="axlearn.experiments.text.gpt.pajama_trainer",
        )()
        record["axlearn"] = {
            "import": "ok",
            "trainer_configs": "7b-sigmoid and 7b-softmax load",
            "install_note": (
                "tensorflow-io==0.37.1 (not 0.37.3) + pip install --no-deps axlearn "
                "+ jax/flax/aqtp manually on Python 3.12"
            ),
        }
        record["corrections_from_exp_032"].append(
            "axlearn loadable on Py3.12 with tensorflow-io 0.37.1 workaround (exp-032 cited Py3.11 only)"
        )
    except Exception as e:
        record["axlearn"] = {"import": f"failed: {e}"}
        record["remaining_blockers"].append(f"axlearn: {e}")

    # Measurement not run
    record["remaining_blockers"].append(
        "Download ~24.5 GB model weights per arm (sigmoid + softmax) from GCS; then InferenceRunner + exp-007 protocol"
    )
    record["next_commands"] = [
        f"mkdir -p research/physics/checkpoints && cd research/physics/checkpoints",
        f"# Follow apple/ml-sigmoid-attention pretrained/axlearn_load_pretrained.ipynb partial copy",
        f"gsutil -m cp -r {SIGMOID_GCS}/tf_* downloads/models/...",
        f"gsutil -m cp -r {SIGMOID_GCS}/gda/model downloads/models/.../gda/",
    ]

    if record["remaining_blockers"] == [
        "Download ~24.5 GB model weights per arm (sigmoid + softmax) from GCS; then InferenceRunner + exp-007 protocol"
    ] and record.get("gcs_access") and record.get("axlearn", {}).get("import") == "ok":
        record["verdict"] = "infrastructure_ready_download_pending"

    RESULTS.write_text(json.dumps(record, indent=2))
    print(json.dumps({"verdict": record["verdict"], "remaining": record["remaining_blockers"][:2]}, indent=2))
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
