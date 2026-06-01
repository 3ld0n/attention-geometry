"""
exp-044: Corrected ALiBi fit for GPT-Neo global layers.

Analysis-only. Load exp-025 BCFT JSON, extract head-level Δ for
global-attention layers only (even-indexed in GPT-Neo-2.7B), apply
coverage filter, and compute the proper head-level Δ distribution.

Pre-stated prediction: global layer head-level Δ_med ≈ 0.10 (trivial
fixed point). The corrected ALiBi interpretation: GPT-Neo full-model
Δ_med=0.126 (exp-039) is confounded by the alternating architecture;
the global-layer population is the clean ALiBi data point, and it sits
near Δ→0, not at SYK q=4 (0.25).

Source experiments: exp-025 (BCFT JSON), exp-040 (per-layer alternating
Δ analysis identifying anomalous layers).
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

# ── Data source ───────────────────────────────────────────────────────────────
BCFT_JSON = Path("research/physics/experiments/exp-025_bcft_pre_registered/results/"
                 "bcft_pre_registered_run_2026-04-17T092239Z.json")
MODEL_KEY = "EleutherAI/gpt-neo-2.7B"

# ── Coverage filter parameters ────────────────────────────────────────────────
# From exp-040: L6 (4/8 conformal, Δ=1.60) and L8 (9/16 conformal, Δ=0.88)
# are flagged as anomalous — few measured heads, outlier Δ values for global type.
# L0, L2 have 0 conformal heads (first two layers, likely not informative).
# L4 has 2/10 conformal heads (very low R² fit rate).
# Reliable criterion: at least MIN_CONFORMAL_HEADS R²>0.90 heads in the layer.
MIN_CONFORMAL_HEADS = 8   # at least 8 conformal heads to count as reliable

# ── GPT-Neo architecture: even layers = global, odd layers = local ─────────────
def layer_type(layer_idx):
    return "global" if layer_idx % 2 == 0 else "local"


def load_gptneo_heads(path, model_key):
    with open(path) as f:
        data = json.load(f)
    model_data = data["models"][model_key]
    heads = model_data["heads"]
    # Each head: {layer, head, delta, R2, valley_depth, start, middle, end}
    return heads, model_data


def main():
    heads, model_data = load_gptneo_heads(BCFT_JSON, MODEL_KEY)

    print(f"Model: {MODEL_KEY}")
    print(f"Total head entries in JSON: {len(heads)}")
    print(f"n_layers: {model_data['n_layers']}, n_heads_per_layer: {model_data['n_heads']}")
    print(f"n_total_heads: {model_data['n_total_heads']}")
    print(f"n_conformal_heads (R²>0.90): {model_data['n_conformal_heads']}")
    print()

    # ── Step 1: Separate global vs local heads ────────────────────────────────
    # All heads in the JSON (not just conformal — full population)
    global_heads_all = [h for h in heads if layer_type(h["layer"]) == "global"]
    local_heads_all  = [h for h in heads if layer_type(h["layer"]) == "local"]

    # Conformal heads only (R² > 0.90, the analysis population)
    R2_THRESHOLD = 0.90
    global_conf = [h for h in global_heads_all if h["R2"] >= R2_THRESHOLD]
    local_conf  = [h for h in local_heads_all  if h["R2"] >= R2_THRESHOLD]

    print(f"Global heads total measured: {len(global_heads_all)}")
    print(f"Global heads R²>0.90: {len(global_conf)}")
    print(f"Local  heads total measured: {len(local_heads_all)}")
    print(f"Local  heads R²>0.90: {len(local_conf)}")
    print()

    # ── Step 2: Per-layer summary for global layers ───────────────────────────
    from collections import defaultdict
    global_layers = sorted(set(h["layer"] for h in global_heads_all))

    print("Per-layer global summary:")
    print(f"{'Layer':>6} {'Measured':>9} {'Conformal':>10} {'Δ_med':>7} {'Reliable?':>10}")
    reliable_layers = []
    anomalous_layers = []

    for L in global_layers:
        measured = [h for h in global_heads_all if h["layer"] == L]
        conformal = [h for h in measured if h["R2"] >= R2_THRESHOLD]
        n_m = len(measured)
        n_c = len(conformal)
        deltas = [h["delta"] for h in conformal]
        d_med = np.median(deltas) if deltas else float("nan")
        reliable = n_c >= MIN_CONFORMAL_HEADS
        flag = "yes" if reliable else ("ANOMALOUS" if n_c > 0 else "no-data")
        print(f"  L{L:<4} {n_m:>8} {n_c:>10} {d_med:>7.3f}   {flag}")
        if reliable:
            reliable_layers.append(L)
        elif n_c > 0:
            anomalous_layers.append(L)
    print()

    # ── Step 3: Head-level Δ for reliable global layers ───────────────────────
    reliable_global_conf = [h for h in global_conf if h["layer"] in reliable_layers]
    delta_reliable = np.array([h["delta"] for h in reliable_global_conf])

    print(f"Reliable global layers: {reliable_layers}")
    print(f"Anomalous global layers (excluded): {anomalous_layers}")
    print(f"Total conformal heads in reliable global layers: {len(delta_reliable)}")
    print()

    if len(delta_reliable) == 0:
        print("ERROR: No reliable global heads found.")
        return

    # ── Step 4: Head-level Δ distribution statistics ──────────────────────────
    d_med = np.median(delta_reliable)
    d_mean = np.mean(delta_reliable)
    d_std = np.std(delta_reliable)
    d_p25, d_p75 = np.percentile(delta_reliable, [25, 75])

    # SYK-near: |Δ - 0.25| ≤ 0.05
    syk_near = delta_reliable[np.abs(delta_reliable - 0.25) <= 0.05]
    # Trivial-near: Δ < 0.15 (within ~60% of trivial fixed point Δ→0)
    trivial_near = delta_reliable[delta_reliable < 0.15]

    print("Head-level Δ distribution (reliable global layers):")
    print(f"  Median Δ:           {d_med:.4f}")
    print(f"  Mean Δ:             {d_mean:.4f}")
    print(f"  Std Δ:              {d_std:.4f}")
    print(f"  25th/75th pctile:   {d_p25:.4f} / {d_p75:.4f}")
    print(f"  SYK-near (|Δ-0.25|≤0.05): {len(syk_near)}/{len(delta_reliable)} ({100*len(syk_near)/len(delta_reliable):.1f}%)")
    print(f"  Trivial-near (Δ<0.15):     {len(trivial_near)}/{len(delta_reliable)} ({100*len(trivial_near)/len(delta_reliable):.1f}%)")
    print()

    # ── Step 5: Depth-stratified analysis ─────────────────────────────────────
    shallow_threshold = 16
    shallow_heads = [h for h in reliable_global_conf if h["layer"] <= shallow_threshold]
    deep_heads    = [h for h in reliable_global_conf if h["layer"] > shallow_threshold]

    d_shallow = np.array([h["delta"] for h in shallow_heads]) if shallow_heads else np.array([])
    d_deep    = np.array([h["delta"] for h in deep_heads]) if deep_heads else np.array([])

    print("Depth-stratified (reliable global layers):")
    if len(d_shallow) > 0:
        print(f"  Shallow (L≤{shallow_threshold}): {len(d_shallow)} heads, Δ_med={np.median(d_shallow):.4f}, mean={np.mean(d_shallow):.4f}")
        sh_layers = sorted(set(h["layer"] for h in shallow_heads))
        print(f"    Layers: {sh_layers}")
    else:
        print(f"  Shallow: no reliable global heads")
    if len(d_deep) > 0:
        print(f"  Deep    (L>{shallow_threshold}): {len(d_deep)} heads, Δ_med={np.median(d_deep):.4f}, mean={np.mean(d_deep):.4f}")
        dp_layers = sorted(set(h["layer"] for h in deep_heads))
        print(f"    Layers: {dp_layers}")
    else:
        print(f"  Deep: no reliable global heads")
    print()

    # ── Step 6: Comparison context ────────────────────────────────────────────
    print("Comparison context:")
    print(f"  Pre-stated prediction:     Δ_med ≈ 0.10 (trivial fixed point)")
    print(f"  Measured (reliable global): Δ_med = {d_med:.4f}")
    delta_diff = abs(d_med - 0.10)
    print(f"  Difference from 0.10:      {delta_diff:.4f}")
    print(f"  SYK q=4 prediction:        0.2500")
    print(f"  GPT-Neo full-model (exp-039): 0.126 (confounded)")
    print(f"  OLMo-7B ALiBi (exp-039):   0.265 (clean ALiBi reference)")
    print()

    # ── Step 7: R² quality check ───────────────────────────────────────────────
    r2_values = np.array([h["R2"] for h in reliable_global_conf])
    print(f"R² distribution for reliable global conformal heads:")
    print(f"  Median R²: {np.median(r2_values):.4f}")
    print(f"  Min R²:    {np.min(r2_values):.4f}")
    print(f"  Heads with R²>0.95: {np.sum(r2_values > 0.95)}/{len(r2_values)}")
    print()

    # ── Step 8: Verdict ───────────────────────────────────────────────────────
    if d_med < 0.15:
        verdict = "CONFIRMED"
        verdict_note = f"Δ_med={d_med:.4f} < 0.15, consistent with trivial fixed point"
    elif d_med < 0.20:
        verdict = "PARTIAL"
        verdict_note = f"Δ_med={d_med:.4f}, above trivial but below SYK range"
    else:
        verdict = "FALSIFIED"
        verdict_note = f"Δ_med={d_med:.4f} does not match trivial fixed point prediction"
    print(f"Verdict: {verdict} — {verdict_note}")
    print()

    # ── Output results JSON ───────────────────────────────────────────────────
    results = {
        "experiment": "exp-044",
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ"),
        "source_experiment": "exp-025",
        "model": MODEL_KEY,
        "architecture_note": "even-index layers = global (full attention); odd = local (window)",
        "coverage_filter": {
            "min_conformal_heads": MIN_CONFORMAL_HEADS,
            "R2_threshold": R2_THRESHOLD,
        },
        "reliable_global_layers": reliable_layers,
        "anomalous_global_layers_excluded": anomalous_layers,
        "n_reliable_global_conformal_heads": int(len(delta_reliable)),
        "delta_stats": {
            "median": float(d_med),
            "mean": float(d_mean),
            "std": float(d_std),
            "p25": float(d_p25),
            "p75": float(d_p75),
        },
        "syk_near_count": int(len(syk_near)),
        "trivial_near_count": int(len(trivial_near)),
        "depth_stratified": {
            "shallow_threshold_layer": shallow_threshold,
            "shallow_n_heads": int(len(d_shallow)),
            "shallow_delta_med": float(np.median(d_shallow)) if len(d_shallow) > 0 else None,
            "deep_n_heads": int(len(d_deep)),
            "deep_delta_med": float(np.median(d_deep)) if len(d_deep) > 0 else None,
        },
        "r2_stats": {
            "median": float(np.median(r2_values)),
            "min": float(np.min(r2_values)),
            "n_above_0p95": int(np.sum(r2_values > 0.95)),
        },
        "pre_stated_prediction": {
            "global_delta_med": 0.10,
            "interpretation": "near trivial fixed point (Δ→0)"
        },
        "comparison": {
            "gptneo_full_model_exp039": 0.126,
            "olmo_7b_alidi_exp039": 0.265,
            "syk_q4_prediction": 0.25,
        },
        "verdict": verdict,
        "verdict_note": verdict_note,
    }

    out_path = Path("research/physics/experiments/exp-044_gptneo_global_alidi_corrected/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
