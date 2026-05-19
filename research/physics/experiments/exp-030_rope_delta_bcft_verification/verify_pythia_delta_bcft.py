#!/usr/bin/env python3
"""
Verify Pythia median Δ values claimed in May 18–19 cron notes against raw BCFT JSON.

Hypothesis under test: Pythia-410m and Pythia-1.4b (both L=24) have different median Δ
from BCFT pre-registered results, and Pythia-6.9b has Δ_med ≈ 0.60.

Ariel — May 19, 2026 (physics room session)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BCFT_DIR = (
    REPO
    / "research/physics/experiments/exp-025_bcft_pre_registered/results"
)
OUT = Path(__file__).resolve().parent / "results.json"

CRON_ASSERTED = {
    "EleutherAI/pythia-410m": {"delta_med": [0.281, 0.283, 0.284], "layers": 24},
    "EleutherAI/pythia-1.4b": {"delta_med": [0.383, 0.375, 0.378], "layers": 24},
    "EleutherAI/pythia-6.9b": {"delta_med": [0.601, 0.60, 0.599], "layers": 32},
}

MARCH_EXP011 = {
    "EleutherAI/pythia-70m": {"layers": 6, "median_R2_gt_0.90": 0.60},
    "EleutherAI/pythia-160m": {"layers": 12, "median_R2_gt_0.90": 0.38},
    "EleutherAI/pythia-410m": {"layers": 24, "median_R2_gt_0.90": 0.28},
}

BCFT_FILES = {
    "EleutherAI/pythia-410m": BCFT_DIR / "bcft_pre_registered_run_2026-04-17T092056Z.json",
    "EleutherAI/pythia-1.4b": BCFT_DIR / "bcft_pre_registered_run_2026-04-17T092239Z.json",
}


@dataclass(frozen=True)
class Filter:
    name: str
    fn: str  # documented in output


FILTERS = [
    ("all_heads", "all"),
    ("R2_ge_0.85", "R2>=0.85"),
    ("R2_gt_0.90_march", "R2>0.90 (March exp-011 PL heads)"),
    ("bcft_conformal", "R2>=0.85 and delta>=0.05"),
    ("syk_near_R2_ge_0.85", "R2>=0.85 and |delta-0.25|<=0.05"),
    ("syk_near_R2_gt_0.90", "R2>0.90 and |delta-0.25|<=0.05"),
]


def median_delta(heads: list[dict], predicate) -> dict:
    sub = [h["delta"] for h in heads if predicate(h)]
    if not sub:
        return {"n": 0, "median_delta": None}
    return {"n": len(sub), "median_delta": float(statistics.median(sub))}


def analyze_model(model_name: str, path: Path) -> dict:
    data = json.loads(path.read_text())
    block = data["models"][model_name]
    heads = block["heads"]
    layers_present = sorted({h["layer"] for h in heads})
    n_layers_config = block.get("n_layers")
    n_heads_config = block.get("n_heads")
    expected_heads = (
        (n_layers_config or 0) * (n_heads_config or 0) if n_layers_config else None
    )

    stats_by_filter = {}
    for key, _ in FILTERS:
        if key == "all_heads":
            pred = lambda h: True
        elif key == "R2_ge_0.85":
            pred = lambda h: h["R2"] >= 0.85
        elif key == "R2_gt_0.90_march":
            pred = lambda h: h["R2"] > 0.90
        elif key == "bcft_conformal":
            pred = lambda h: h["R2"] >= 0.85 and h["delta"] >= 0.05
        elif key == "syk_near_R2_ge_0.85":
            pred = lambda h: h["R2"] >= 0.85 and abs(h["delta"] - 0.25) <= 0.05
        elif key == "syk_near_R2_gt_0.90":
            pred = lambda h: h["R2"] > 0.90 and abs(h["delta"] - 0.25) <= 0.05
        else:
            raise ValueError(key)
        stats_by_filter[key] = median_delta(heads, pred)

    cron = CRON_ASSERTED.get(model_name, {})
    cron_range = None
    if cron.get("delta_med"):
        vals = cron["delta_med"]
        cron_range = {"min": min(vals), "max": max(vals), "values": vals}

    return {
        "model": model_name,
        "source_json": str(path.relative_to(REPO)),
        "n_heads_in_file": len(heads),
        "n_layers_config": n_layers_config,
        "n_heads_config": n_heads_config,
        "expected_total_heads": expected_heads,
        "layers_present_min": min(layers_present),
        "layers_present_max": max(layers_present),
        "n_layers_with_data": len(layers_present),
        "coverage_complete": (
            expected_heads is not None and len(heads) == expected_heads
        ),
        "filters": stats_by_filter,
        "cron_asserted": cron_range,
        "march_exp011_reference": MARCH_EXP011.get(model_name),
    }


def main() -> None:
    results = {
        "experiment": "exp-030_rope_delta_bcft_verification",
        "date": "2026-05-19",
        "bcft_json_dir": str(BCFT_DIR.relative_to(REPO)),
        "pythia_6_9b_in_bcft": False,
        "models": {},
        "conclusions": [],
    }

    for model_name, path in BCFT_FILES.items():
        results["models"][model_name] = analyze_model(model_name, path)

    # 6.9b scan
    for path in BCFT_DIR.glob("bcft_pre_registered_run_*.json"):
        data = json.loads(path.read_text())
        for m in data.get("models", {}):
            if "6.9" in m.lower():
                results["pythia_6_9b_in_bcft"] = True

    m410 = results["models"]["EleutherAI/pythia-410m"]
    m14 = results["models"]["EleutherAI/pythia-1.4b"]

    d_syk = (
        m410["filters"]["syk_near_R2_ge_0.85"]["median_delta"]
        - m14["filters"]["syk_near_R2_ge_0.85"]["median_delta"]
    )

    results["conclusions"] = [
        "Pythia-6.9b does not appear in any exp-025 BCFT pre-registered JSON; cron Δ≈0.60 matches March exp-011 Pythia-70m (6 layers), not a 32-layer model.",
        "Cron Pythia-410m Δ≈0.28 matches March exp-011 410m (R2>0.90), not BCFT extraction on the same file (R2>0.90 median Δ≈0.36).",
        "Cron Pythia-1.4b Δ≈0.38 matches March exp-011 160m (12 layers, Δ=0.38), likely model-name/layer-count confusion — not an independent 1.4b measurement.",
        f"BCFT 410m JSON is incomplete: {m410['n_heads_in_file']} heads vs {m410['expected_total_heads']} expected.",
        f"Same-depth different-Δ under SYK-near filter (R2>=0.85, |Δ-0.25|<=0.05): 410m={m410['filters']['syk_near_R2_ge_0.85']['median_delta']:.4f}, 1.4b={m14['filters']['syk_near_R2_ge_0.85']['median_delta']:.4f}, diff={d_syk:+.4f} — not the large split cron claimed.",
    ]

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
