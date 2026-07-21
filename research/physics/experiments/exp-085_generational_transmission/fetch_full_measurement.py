"""
Fetch the full exp-085 measurement JSON (per-head data) from the Modal volume
and save it locally for figure-making and per-head reporting.

    .venv/bin/python3 -m modal run research/physics/experiments/exp-085_generational_transmission/fetch_full_measurement.py

Ariel — July 21, 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
app = modal.App("exp085-fetch")
vol_085 = modal.Volume.from_name("exp085-data", create_if_missing=False)

RUN_NAME = "run_Cgen_s0"


@app.function(timeout=120, volumes={"/data085": vol_085}, memory=4096)
def fetch() -> dict:
    result_path = Path(f"/data085/measurements/{RUN_NAME}.json")
    if not result_path.exists():
        return {"error": f"{result_path} not found"}
    return json.loads(result_path.read_text())


@app.local_entrypoint()
def main():
    data = fetch.remote()
    if "error" in data:
        raise SystemExit(data["error"])
    out = SCRIPT_DIR / "measurements"
    out.mkdir(exist_ok=True)
    path = out / f"{RUN_NAME}_final.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"Wrote {path} ({len(data.get('heads', []))} per-head records)")
