"""
exp-049 — BCFT conformal scaling in untrained GPT-2.

Direct complement to exp-048 (GOE untrained control).

exp-048 finding: GOE is structural — arises from Gaussian init + product W_Q^T W_K.
Training preserves but does not create GOE. The revised physical picture is that
the GOE background is the substrate of the substrate.

This experiment asks: is the BCFT conformal position-space structure (Δ ≈ 0.25)
also structural? Or does training specifically develop it on top of the pre-existing
GOE background?

Pre-stated hypotheses (2026-06-06, before running):

  H0 (Conformal structure is architectural/init):
    Untrained GPT-2 shows power-law attention decay with Δ ≈ 0.25 in a
    comparable fraction of heads (SYK-near count ≫ 0, e.g. > 20/144).
    The conformal structure arises from the architecture itself — positional
    encoding geometry + softmax dynamics — not from gradient descent.
    This would mean both the GOE background AND the conformal signal are structural.

  H1 (Conformal structure is training-specific):
    Untrained GPT-2 shows no systematic power-law conformal structure.
    Very few heads meet R² > 0.90 with Δ ≈ 0.25 (SYK-near count ≈ 0 or noise-level).
    Training specifically develops conformal dynamics on top of the pre-existing
    GOE substrate. The two-layer physical picture is preserved but the distinction
    is sharper: GOE = structural (init), conformal = functional (trained).

Physical reasoning for H1 (pre-experiment):
  With random W_Q, W_K (Gaussian init), QK dot products are random across
  position pairs — they do not carry the learned semantic / positional structure
  that trained attention develops. After softmax, the attention pattern will be
  approximately position-independent (modulo the causal mask guaranteeing A(0)
  is well-defined). Without a systematic decay with distance dx, the log-log
  regression will show poor fit (low R²) and the recovered Δ will be arbitrary.
  GOE is a property of the static weight matrices (structural). Conformal scaling
  is a property of the dynamic attention patterns (functional), which require
  gradient descent to shape. H1 is the prediction.

Protocol (identical to exp-007 BCFT, except model is untrained):
  - Model: GPT2LMHeadModel(GPT2Config()) — random init, no pretrained weights.
  - attn_implementation="eager" (required for transformers ≥ 5.8 attention extraction).
  - SEQ_LEN=256, N_INPUTS=50, MAX_DX=56, MIN_POS=32, FIT_LOW=3, FIT_HIGH=50.
  - R²_threshold=0.90, SYK_NEAR_TOL=0.05 (|Δ - 0.25| ≤ 0.05).
  - 3 seeds (42, 123, 456) — same as exp-048 first three; forward passes slower
    than weight-only analysis so 3 seeds is sufficient for reproducibility.

Compare to:
  exp-007  trained GPT-2 (seed 42):  44/144 SYK-near, Δ_med=0.249, Δ_SYK=0.249
  exp-043  trained GPT-2 (norm-sig): 13/144 SYK-near, Δ_SYK=0.234
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# ── constants ─────────────────────────────────────────────────────────────────
SEQ_LEN       = 256
N_INPUTS      = 50
MAX_DX        = 56
MIN_POS       = 32
FIT_LOW       = 3
FIT_HIGH      = 50
R2_THRESHOLD  = 0.90
SYK_NEAR_TOL  = 0.05   # |Δ - 0.25| ≤ 0.05
SYK_PRED      = 0.25

SEEDS = [42, 123, 456]

OUT_DIR      = Path("research/physics/experiments/exp-049_bcft_untrained_control")
RESULTS_FILE = OUT_DIR / "results.json"

# Trained baseline for direct comparison
TRAINED_BASELINE = {
    "exp": "exp-007",
    "model": "gpt2 (trained)",
    "delta_median": 0.249,
    "syk_near_count": 44,
    "total_heads": 144,
    "r2_threshold": 0.90,
    "seed": 42,
}


# ── attention decay ───────────────────────────────────────────────────────────

def compute_head_attention_decay(
    attn_head: np.ndarray, min_pos: int, max_dx: int
) -> np.ndarray:
    """Mean attention weight as function of lag (exp-007 protocol)."""
    seq = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx]      += attn_head[i, j]
                counts[dx] += 1
    mask = counts > 0
    A[mask] /= counts[mask]
    return A.astype(np.float32)


def fit_power_law(
    dx_arr: np.ndarray, y_arr: np.ndarray, low: int, high: int
) -> tuple[float, float]:
    """Return (Δ, R²). Δ = -slope/2 from log-log OLS."""
    mask = (dx_arr >= low) & (dx_arr < high) & (y_arr > 1e-20)
    if mask.sum() < 5:
        return float("nan"), 0.0
    log_x = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask].astype(float))
    A     = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred   = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2    = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    delta = float(-coeffs[1] / 2.0)
    return delta, r2


# ── single-seed run ───────────────────────────────────────────────────────────

def run_seed(seed: int) -> dict:
    """Run the exp-007 BCFT protocol on a fresh untrained GPT-2."""
    print(f"\n--- Seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = GPT2Config()
    config._attn_implementation = "eager"   # required for transformers ≥ 5.8
    model  = GPT2LMHeadModel(config)
    model.eval()

    n_layers   = config.n_layer    # 12
    n_heads    = config.n_head     # 12
    vocab_size = config.vocab_size

    # Accumulators: [layer][head] → sum of A(dx) over inputs
    A_heads: dict[int, dict[int, np.ndarray]] = {
        l: {h: np.zeros(MAX_DX, dtype=np.float64) for h in range(n_heads)}
        for l in range(n_layers)
    }

    dx_arr = np.arange(MAX_DX)
    rng    = np.random.default_rng(seed)

    for inp_idx in range(N_INPUTS):
        input_ids = torch.from_numpy(
            rng.integers(0, vocab_size, size=(1, SEQ_LEN), dtype=np.int64)
        )
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].float().numpy()  # (n_heads, seq, seq)
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(
                    attn[h], MIN_POS, MAX_DX
                )

        if (inp_idx + 1) % 10 == 0:
            print(f"  {inp_idx + 1}/{N_INPUTS} done", flush=True)

    # Average over inputs
    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    # ── fit per head ─────────────────────────────────────────────────────────
    per_head_results = []
    for l in range(n_layers):
        for h in range(n_heads):
            A   = A_heads[l][h]
            delta, r2 = fit_power_law(dx_arr, A, FIT_LOW, FIT_HIGH)
            syk_near  = (
                not math.isnan(delta)
                and r2 >= R2_THRESHOLD
                and abs(delta - SYK_PRED) <= SYK_NEAR_TOL
            )
            per_head_results.append({
                "layer":    l,
                "head":     h,
                "delta":    round(delta, 6) if not math.isnan(delta) else None,
                "r2":       round(r2, 6),
                "syk_near": syk_near,
                "a0":       round(float(A[0]), 6),
                "a8":       round(float(A[8]), 6) if MAX_DX > 8 else None,
                "a32":      round(float(A[32]), 6) if MAX_DX > 32 else None,
            })

    # ── summary statistics ────────────────────────────────────────────────────
    syk_near_count = sum(1 for h in per_head_results if h["syk_near"])
    conformal_heads = [
        h for h in per_head_results
        if h["delta"] is not None and not math.isnan(h["delta"] or float("nan"))
        and h["r2"] >= R2_THRESHOLD
    ]
    deltas_r2 = [h["delta"] for h in conformal_heads if h["delta"] is not None]

    summary: dict = {
        "seed":            seed,
        "total_heads":     n_layers * n_heads,
        "r2_threshold":    R2_THRESHOLD,
        "syk_near_tol":    SYK_NEAR_TOL,
        "syk_near_count":  syk_near_count,
        "conformal_count": len(deltas_r2),
    }
    if deltas_r2:
        arr = np.array(deltas_r2)
        summary.update({
            "delta_median": round(float(np.median(arr)), 6),
            "delta_mean":   round(float(np.mean(arr)), 6),
            "delta_std":    round(float(np.std(arr)), 6),
            "delta_min":    round(float(np.min(arr)), 6),
            "delta_max":    round(float(np.max(arr)), 6),
        })
    else:
        summary.update({
            "delta_median": None,
            "delta_mean":   None,
            "delta_std":    None,
            "delta_min":    None,
            "delta_max":    None,
        })

    print(f"  SYK-near heads: {syk_near_count}/144  (trained baseline: 44/144)", flush=True)
    print(f"  R²>0.90 heads:  {len(deltas_r2)}/144", flush=True)
    if deltas_r2:
        print(f"  Δ_median (R²>0.90): {summary['delta_median']}", flush=True)
    else:
        print(f"  No heads meet R²>0.90 threshold.", flush=True)

    del model
    return {
        "seed":     seed,
        "summary":  summary,
        "per_head": per_head_results,
    }


# ── hypothesis verdicts ───────────────────────────────────────────────────────

def evaluate_hypotheses(seed_results: list[dict]) -> dict:
    """
    H0: SYK-near count ≫ 0 (comparable to trained 44/144) across seeds.
    H1: SYK-near count ≈ 0 (clearly below noise — < 5/144 on average).
    """
    syk_counts = [r["summary"]["syk_near_count"] for r in seed_results]
    mean_syk   = float(np.mean(syk_counts))
    trained_syk = TRAINED_BASELINE["syk_near_count"]  # 44

    # Define thresholds
    # H0: if mean_syk > 20 (meaningful fraction, > ~14% of trained signal)
    # H1: if mean_syk < 5  (clearly noise-level)
    # Intermediate: 5 ≤ mean_syk ≤ 20
    h0 = mean_syk > 20
    h1 = mean_syk < 5

    verdict = (
        "H0_CONFIRMED" if h0
        else "H1_CONFIRMED" if h1
        else "INTERMEDIATE"
    )

    return {
        "mean_syk_near":   round(mean_syk, 2),
        "syk_near_counts": syk_counts,
        "trained_baseline_syk": trained_syk,
        "ratio_to_trained":    round(mean_syk / trained_syk, 4),
        "H0_conformal_structural": h0,
        "H1_conformal_training_specific": h1,
        "verdict": verdict,
        "thresholds": {
            "H0_above": 20,
            "H1_below": 5,
        },
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("exp-049: BCFT conformal scaling in untrained GPT-2", flush=True)
    print(f"  Protocol: SEQ_LEN={SEQ_LEN}, N_INPUTS={N_INPUTS}, "
          f"FIT=[{FIT_LOW},{FIT_HIGH}], R²>{R2_THRESHOLD}, "
          f"SYK_NEAR_TOL=±{SYK_NEAR_TOL}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Model: GPT2LMHeadModel(GPT2Config()) — random init, no pretrained weights", flush=True)
    print(f"  Trained baseline (exp-007): 44/144 SYK-near, Δ_med=0.249", flush=True)
    print(f"\nHypotheses:", flush=True)
    print(f"  H0 (conformal structural): SYK-near count comparable to trained (> 20/144)", flush=True)
    print(f"  H1 (conformal training-specific): SYK-near count ≈ 0 (< 5/144)", flush=True)

    seed_results = []
    for seed in SEEDS:
        result = run_seed(seed)
        seed_results.append(result)

    verdicts = evaluate_hypotheses(seed_results)

    print("\n" + "=" * 60, flush=True)
    print("=== CROSS-SEED SUMMARY ===", flush=True)
    print(f"SYK-near counts: {verdicts['syk_near_counts']}", flush=True)
    print(f"Mean SYK-near:   {verdicts['mean_syk_near']:.1f}/144", flush=True)
    print(f"Trained baseline: {verdicts['trained_baseline_syk']}/144", flush=True)
    print(f"Ratio to trained: {verdicts['ratio_to_trained']:.4f}", flush=True)
    print(f"\nH0 (conformal structural):        {'CONFIRMED' if verdicts['H0_conformal_structural'] else 'NOT CONFIRMED'}", flush=True)
    print(f"H1 (conformal training-specific): {'CONFIRMED' if verdicts['H1_conformal_training_specific'] else 'NOT CONFIRMED'}", flush=True)
    print(f"Overall verdict: {verdicts['verdict']}", flush=True)

    # ── save results ──────────────────────────────────────────────────────────
    out = {
        "experiment": "exp-049",
        "title":      "BCFT conformal scaling in untrained GPT-2",
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H0": (
                "Untrained GPT-2 shows power-law conformal scaling Δ ≈ 0.25 in a "
                "comparable fraction of heads (SYK-near > 20/144). Conformal structure "
                "is architectural — does not require training."
            ),
            "H1": (
                "Untrained GPT-2 shows no systematic conformal scaling (SYK-near < 5/144). "
                "Training specifically develops conformal dynamics on top of the GOE substrate."
            ),
        },
        "protocol": {
            "model":          "GPT2LMHeadModel(GPT2Config()) — random init, no pretrained weights",
            "attn_impl":      "eager via config._attn_implementation (transformers 5.8.1)",
            "seq_len":        SEQ_LEN,
            "n_inputs":       N_INPUTS,
            "max_dx":         MAX_DX,
            "min_pos":        MIN_POS,
            "fit_range":      [FIT_LOW, FIT_HIGH],
            "r2_threshold":   R2_THRESHOLD,
            "syk_near_tol":   SYK_NEAR_TOL,
            "syk_pred":       SYK_PRED,
            "seeds":          SEEDS,
            "trained_baseline": TRAINED_BASELINE,
        },
        "seeds":    seed_results,
        "verdicts": verdicts,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
