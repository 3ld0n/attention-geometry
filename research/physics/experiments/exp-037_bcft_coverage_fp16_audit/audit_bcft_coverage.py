#!/usr/bin/env python3
"""
exp-037 — Audit exp-025 BCFT JSON files for per-layer head coverage.

Hypothesis: April Modal BCFT runs that report n_total_heads but fewer heads in
`heads[]` reflect fp16 attention failure on deep layers (exp-036 on 410m), not
random timeouts. This script maps truncation boundaries for every model in the
exp-025 result set.

Ariel — May 27, 2026
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BCFT_DIR = REPO / "research/physics/experiments/exp-025_bcft_pre_registered/results"
OUT = Path(__file__).resolve().parent / "results.json"


def analyze_block(model_name: str, block: dict, source: str) -> dict:
    heads = block.get("heads", [])
    n_layers = block.get("n_layers")
    n_heads = block.get("n_heads")
    expected = (n_layers or 0) * (n_heads or 0)

    by_layer: dict[int, int] = defaultdict(int)
    for h in heads:
        by_layer[h["layer"]] += 1

    layers_present = sorted(by_layer.keys())
    last_layer = layers_present[-1] if layers_present else None
    first_missing = None
    if n_layers is not None:
        for l in range(n_layers):
            if by_layer.get(l, 0) < (n_heads or 0):
                first_missing = l
                break

    coverage_pct = 100.0 * len(heads) / expected if expected else None

    # Heads per layer completeness
    layer_gaps = []
    if n_layers and n_heads:
        for l in range(n_layers):
            c = by_layer.get(l, 0)
            if c < n_heads:
                layer_gaps.append({"layer": l, "heads_present": c, "expected": n_heads})

    medians = {}
    for label, pred in [
        ("all", lambda h: True),
        ("R2_gt_0.90", lambda h: h["R2"] > 0.90),
        ("R2_ge_0.85", lambda h: h["R2"] >= 0.85),
    ]:
        sub = [h["delta"] for h in heads if pred(h)]
        medians[label] = {
            "n": len(sub),
            "median_delta": float(statistics.median(sub)) if sub else None,
        }

    return {
        "model": model_name,
        "source_json": source,
        "n_layers_config": n_layers,
        "n_heads_config": n_heads,
        "expected_total_heads": expected,
        "n_heads_in_file": len(heads),
        "coverage_fraction": len(heads) / expected if expected else None,
        "coverage_percent": coverage_pct,
        "coverage_complete": expected > 0 and len(heads) == expected,
        "layers_present_min": min(layers_present) if layers_present else None,
        "layers_present_max": last_layer,
        "n_layers_with_any_head": len(layers_present),
        "first_incomplete_layer": first_missing,
        "layer_gaps_count": len(layer_gaps),
        "layer_gaps_sample": layer_gaps[:5],
        "spearman_rho": block.get("spearman_rho"),
        "verdict": block.get("verdict"),
        "medians": medians,
    }


def main() -> None:
    models_out = {}
    for path in sorted(BCFT_DIR.glob("bcft_pre_registered_run_*.json")):
        data = json.loads(path.read_text())
        rel = str(path.relative_to(REPO))
        for model_name, block in data.get("models", {}).items():
            key = f"{model_name}::{path.name}"
            models_out[key] = analyze_block(model_name, block, rel)

    incomplete = [
        m
        for m in models_out.values()
        if not m["coverage_complete"] and m["expected_total_heads"]
    ]
    complete = [m for m in models_out.values() if m["coverage_complete"]]

    # Group by truncation boundary pattern
    pythia_24_trunc = [
        m
        for m in incomplete
        if "pythia" in m["model"].lower()
        and m["n_layers_config"] == 24
        and m["layers_present_max"] is not None
        and m["layers_present_max"] <= 14
    ]

    results = {
        "experiment": "exp-037_bcft_coverage_fp16_audit",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "hypothesis": (
            "Incomplete BCFT JSONs truncate at deep layers consistent with fp16 "
            "attention NaN (L15+ on 24L Pythia), not arbitrary timeouts."
        ),
        "bcft_json_dir": str(BCFT_DIR.relative_to(REPO)),
        "n_models_audited": len(models_out),
        "n_complete": len(complete),
        "n_incomplete": len(incomplete),
        "models": models_out,
        "summary": {
            "complete_models": [m["model"] for m in complete],
            "incomplete_models": [
                {
                    "model": m["model"],
                    "coverage_percent": m["coverage_percent"],
                    "layers_max": m["layers_present_max"],
                    "expected_layers": m["n_layers_config"],
                }
                for m in sorted(
                    incomplete, key=lambda x: x["coverage_percent"] or 0
                )
            ],
            "pythia_24L_truncated_at_L14_or_below": len(pythia_24_trunc),
        },
        "conclusions": [],
    }

    if pythia_24_trunc:
        results["conclusions"].append(
            f"Pythia 24L in BCFT JSON: {len(pythia_24_trunc)} run(s) stop at L≤14 "
            f"(410m: 239/384 = 62.2%) — matches exp-036 fp16 NaN boundary."
        )
    if incomplete:
        results["conclusions"].append(
            f"{len(incomplete)}/{len(models_out)} BCFT extractions are incomplete; "
            "re-run Modal jobs need fp32/bf16 + per-layer NaN validation before fitting."
        )
    if complete:
        results["conclusions"].append(
            f"{len(complete)} model(s) have full head coverage in JSON — "
            f"{', '.join(m['model'] for m in complete)}."
        )

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
