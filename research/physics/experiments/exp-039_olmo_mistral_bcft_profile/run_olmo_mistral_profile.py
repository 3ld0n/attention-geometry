#!/usr/bin/env python3
"""
exp-039 — OLMo-7B (ALiBi) and Mistral-7B (RoPE+SWA) per-layer BCFT Δ profile.

Analysis-only: loads existing exp-025 BCFT JSONs (no model run).
OLMo-7B-hf: 100% coverage (allenai/OLMo-7B-hf, ALiBi PE).
Mistral-7B-v0.3: 100% coverage (RoPE + sliding-window attention).
GPT-Neo-2.7B: 89.4% coverage (ALiBi PE) — secondary check.

Pre-stated hypothesis (before looking at numbers):
  1. OLMo-7B (ALiBi): Δ_med (R²>0.90) > GPT-2 learned-PE 0.249. Expected 0.30–0.45.
     Basis: exp-029 ALiBi toy showed ALiBi elevates apparent Δ.
  2. Mistral-7B (RoPE+SWA): Δ_med > 0.249, near Pythia-410m BCFT 0.358 if RoPE
     dominates; SWA may depress deep-layer Δ.
  3. GPT-Neo (ALiBi, partial): tracks OLMo direction if ALiBi elevation is consistent.

Protocol: R²>0.90 filter for global and per-layer Δ_med (exp-007/031/036 standard).
Band split: shallow (L0–10), mid (L11–21), deep (L22–31) for 32-layer models.

References:
  exp-007: GPT-2 (learned PE, 12L) Δ_med = 0.2493
  exp-031: GPT-2-medium (learned PE, 24L) Δ_med = 0.2589
  exp-031: GPT-2-large (learned PE, 36L) Δ_med = 0.2819
  exp-036: Pythia-410m BCFT (RoPE, 24L, fp32) Δ_med = 0.358

Ariel — May 28, 2026
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
BCFT_DIR = Path("research/physics/experiments/exp-025_bcft_pre_registered/results")
FILE2 = BCFT_DIR / "bcft_pre_registered_run_2026-04-17T092239Z.json"
FILE3 = BCFT_DIR / "bcft_pre_registered_run_2026-04-17T095022Z.json"
OUT = Path("research/physics/experiments/exp-039_olmo_mistral_bcft_profile/results.json")

# ── references ────────────────────────────────────────────────────────────────
REFERENCES = {
    "gpt2_small_exp007": {"delta_med": 0.2493, "n_layers": 12, "pos_enc": "learned"},
    "gpt2_medium_exp031": {"delta_med": 0.2589, "n_layers": 24, "pos_enc": "learned"},
    "gpt2_large_exp031": {"delta_med": 0.2819, "n_layers": 36, "pos_enc": "learned"},
    "pythia_410m_exp036_bcft": {"delta_med": 0.358, "n_layers": 24, "pos_enc": "RoPE"},
}

R2_THRESH = 0.90
SYK_WINDOW = 0.05  # |Δ - 0.25| ≤ 0.05


def analyze_model(heads: list[dict], n_layers: int, model_name: str) -> dict:
    """Compute global and per-layer Δ stats using R²>0.90 filter."""
    deltas = np.array([h["delta"] for h in heads])
    r2s = np.array([h["R2"] for h in heads])
    layers = np.array([h["layer"] for h in heads])

    mask = r2s >= R2_THRESH
    syk_mask = mask & (np.abs(deltas - 0.25) <= SYK_WINDOW)

    n_good = int(mask.sum())
    delta_med = float(np.median(deltas[mask])) if n_good > 0 else float("nan")
    n_syk = int(syk_mask.sum())

    # per-layer medians (R²>0.90 heads only)
    per_layer = []
    for lyr in range(n_layers):
        lmask = mask & (layers == lyr)
        if lmask.sum() > 0:
            per_layer.append({
                "layer": lyr,
                "n_good": int(lmask.sum()),
                "delta_med": float(np.median(deltas[lmask])),
            })
        else:
            per_layer.append({"layer": lyr, "n_good": 0, "delta_med": None})

    # band analysis (thirds for 32-layer models)
    band_lo = n_layers // 3
    band_hi = 2 * n_layers // 3
    bands = {
        "shallow": {"range": [0, band_lo - 1]},
        "mid":     {"range": [band_lo, band_hi - 1]},
        "deep":    {"range": [band_hi, n_layers - 1]},
    }
    for bname, bval in bands.items():
        lo, hi = bval["range"]
        bmask = mask & (layers >= lo) & (layers <= hi)
        bval["n_good"] = int(bmask.sum())
        bval["delta_med"] = float(np.median(deltas[bmask])) if bmask.sum() > 0 else None

    return {
        "model": model_name,
        "n_total_heads": len(heads),
        "n_good_R2_90": n_good,
        "frac_good": round(n_good / len(heads), 4) if heads else 0.0,
        "delta_med_global": round(delta_med, 4),
        "delta_vs_syk": round(delta_med - 0.25, 4),
        "n_syk_near": n_syk,
        "frac_syk": round(n_syk / len(heads), 4),
        "per_layer": per_layer,
        "bands": bands,
    }


def load_model_heads(fpath: Path, model_key: str) -> tuple[list[dict], int]:
    with open(fpath) as f:
        data = json.load(f)
    m = data["models"][model_key]
    return m["heads"], m["n_layers"]


def main() -> None:
    results: dict = {
        "experiment": "exp-039_olmo_mistral_bcft_profile",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "R2_threshold": R2_THRESH,
        "SYK_window": SYK_WINDOW,
        "hypothesis": {
            "OLMo_7B_ALiBi": "Δ_med (R²>0.90) > 0.249 (GPT-2 learned PE); expected 0.30–0.45 from exp-029 ALiBi toy",
            "Mistral_7B_RoPE_SWA": "Δ_med > 0.249; near Pythia-410m BCFT 0.358 if RoPE dominates; SWA may depress deep layers",
            "GPT_Neo_ALiBi_partial": "tracks OLMo direction if ALiBi elevation consistent",
        },
        "references": REFERENCES,
        "models": {},
    }

    # OLMo-7B-hf (ALiBi, 100% coverage)
    print("Analyzing OLMo-7B-hf...", flush=True)
    heads, n_layers = load_model_heads(FILE2, "allenai/OLMo-7B-hf")
    olmo_result = analyze_model(heads, n_layers, "allenai/OLMo-7B-hf")
    olmo_result["pos_enc"] = "ALiBi"
    olmo_result["coverage_pct"] = 100.0
    results["models"]["OLMo-7B"] = olmo_result
    print(f"  OLMo Δ_med={olmo_result['delta_med_global']:.4f}  n_R2>0.9={olmo_result['n_good_R2_90']}/{olmo_result['n_total_heads']}")

    # Mistral-7B-v0.3 (RoPE+SWA, 100% coverage)
    print("Analyzing Mistral-7B-v0.3...", flush=True)
    heads, n_layers = load_model_heads(FILE3, "mistralai/Mistral-7B-v0.3")
    mistral_result = analyze_model(heads, n_layers, "mistralai/Mistral-7B-v0.3")
    mistral_result["pos_enc"] = "RoPE+SWA"
    mistral_result["coverage_pct"] = 100.0
    results["models"]["Mistral-7B"] = mistral_result
    print(f"  Mistral Δ_med={mistral_result['delta_med_global']:.4f}  n_R2>0.9={mistral_result['n_good_R2_90']}/{mistral_result['n_total_heads']}")

    # GPT-Neo-2.7B (ALiBi, 89.4% coverage — secondary)
    print("Analyzing GPT-Neo-2.7B (secondary, 89.4% coverage)...", flush=True)
    heads, n_layers = load_model_heads(FILE2, "EleutherAI/gpt-neo-2.7B")
    neo_result = analyze_model(heads, n_layers, "EleutherAI/gpt-neo-2.7B")
    neo_result["pos_enc"] = "ALiBi"
    neo_result["coverage_pct"] = 89.4
    neo_result["coverage_note"] = "partial — L0-31 present but 89.4% of heads; deep-layer band may be incomplete"
    results["models"]["GPT-Neo-2.7B"] = neo_result
    print(f"  GPT-Neo Δ_med={neo_result['delta_med_global']:.4f}  n_R2>0.9={neo_result['n_good_R2_90']}/{neo_result['n_total_heads']} ({neo_result['coverage_pct']}%)")

    # Verdict
    olmo_delta = olmo_result["delta_med_global"]
    mistral_delta = mistral_result["delta_med_global"]
    gpt2_baseline = 0.2493
    pythia_bcft = 0.358

    olmo_elevated = olmo_delta > gpt2_baseline
    mistral_near_pythia = abs(mistral_delta - pythia_bcft) < abs(mistral_delta - gpt2_baseline)

    results["verdict"] = {
        "OLMo_ALiBi_elevated_vs_GPT2": olmo_elevated,
        "OLMo_delta_med": olmo_delta,
        "Mistral_RoPE_SWA_nearer_Pythia_than_GPT2": mistral_near_pythia,
        "Mistral_delta_med": mistral_delta,
        "GPT_Neo_ALiBi_delta_med": neo_result["delta_med_global"],
        "hypothesis_1_OLMo": "confirmed" if olmo_elevated and olmo_delta > 0.30 else (
            "partial" if olmo_elevated else "falsified"
        ),
        "hypothesis_2_Mistral": "confirmed" if mistral_near_pythia else "falsified",
        "notes": "",
    }

    # Print summary table
    print()
    print("─" * 60)
    print(f"{'Model':<30} {'PE':<12} {'Δ_med':>6} {'n_good/tot':>12}")
    print("─" * 60)
    for label, r in [
        ("GPT-2 (exp-007, ref)", "learned"),
        ("GPT-2-medium (exp-031, ref)", "learned"),
        ("Pythia-410m BCFT (exp-036, ref)", "RoPE"),
    ]:
        ref = REFERENCES.get(
            {"GPT-2 (exp-007, ref)": "gpt2_small_exp007",
             "GPT-2-medium (exp-031, ref)": "gpt2_medium_exp031",
             "Pythia-410m BCFT (exp-036, ref)": "pythia_410m_exp036_bcft"}[label]
        )
        print(f"  {label:<28} {ref['pos_enc']:<12} {ref['delta_med']:>6.4f}")
    for name, res in results["models"].items():
        n = f"{res['n_good_R2_90']}/{res['n_total_heads']}"
        print(f"  {name:<28} {res['pos_enc']:<12} {res['delta_med_global']:>6.4f} {n:>12}")
    print("─" * 60)

    print()
    print("Band analysis (32L models, thirds: shallow L0–10, mid L11–21, deep L22–31):")
    for name, res in results["models"].items():
        b = res["bands"]
        sh = b["shallow"]["delta_med"]
        mi = b["mid"]["delta_med"]
        de = b["deep"]["delta_med"]
        sh_s = f"{sh:.4f}" if sh is not None else "nan"
        mi_s = f"{mi:.4f}" if mi is not None else "nan"
        de_s = f"{de:.4f}" if de is not None else "nan"
        print(f"  {name}: shallow={sh_s}  mid={mi_s}  deep={de_s}")

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
