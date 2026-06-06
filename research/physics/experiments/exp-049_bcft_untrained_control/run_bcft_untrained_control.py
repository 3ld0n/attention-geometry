"""
exp-049 — BCFT conformal scaling in untrained GPT-2

This is the direct complement to exp-048 (GOE untrained control).
exp-048 showed: GOE is structural — arises from Gaussian init + product matrix
structure, present before any training. Training preserves but does not create GOE.

The remaining question: does conformal POSITION-SPACE structure (power-law attention
decay with Δ ≈ 0.25) also arise from initialization, or does it require training?

These two measurements are physically distinct:
  - exp-048 measures W_QK eigenvalue statistics (weight-space geometry, no forward pass)
  - exp-049 measures attention decay A(dx) from forward passes (position-space geometry)

Pre-stated hypotheses (before running):

  H1 (Training-dependent conformal structure):
    Untrained GPT-2 produces ~0 SYK-near heads (R²>0.90, |Δ-0.25|≤0.05).
    Conformal position-space structure requires training — random attention weights
    produce near-uniform attention (softmax of tiny logits with σ_init=0.02), which
    gives no structured power-law decay.
    Physical consequence: training develops conformal dynamics on top of a pre-existing
    structural GOE background. Two-layer picture is clean: structural chaos (GOE) at
    init; conformal signal learned through gradient descent.

  H0 (Structural conformal):
    Untrained GPT-2 shows substantial SYK-near count, comparable to trained (44/144).
    Conformal position-space structure also arises from architecture + initialization.
    Physical consequence: would require understanding why random attention exhibits
    power-law decay. More surprising — would revise the two-layer picture significantly.

Physical reasoning for H1:
  GPT-2 default init: nn.init.normal_(weight, std=0.02).
  QK logits: q_i^T k_j / sqrt(d_k) with d_k=64, variance ≈ d_k × σ² = 64 × 0.0004 = 0.026.
  Softmax of near-zero logits → near-uniform distribution over causal context.
  Near-uniform causal attention: A[dx] decays only due to sampling (fewer eligible
  positions for large dx), producing a slow arithmetic-mean decay, not a power law.
  Expected: few R²>0.90 heads; those that appear have random, not clustered, Δ values.

Protocol: identical to exp-007/exp-043 BCFT protocol:
  - SEQ_LEN=256, N_INPUTS=50, FIT=[3,50], R²>0.90, SYK_NEAR_TOL=0.05
  - attn_implementation="eager" (required for transformers ≥5.8)
  - GPT-2 from random init: GPT2LMHeadModel(GPT2Config()) — no pretrained weights
  - 3 seeds (42, 123, 456) to assess whether any signal varies with init

Compare to:
  exp-007  GPT-2 trained softmax:   44/144 SYK-near, Δ_med=0.2493, Δ_SYK=0.2493
  exp-043  GPT-2 trained norm-sig:  13/144 SYK-near, Δ_med=0.2340, Δ_SYK=0.2340
  exp-048  GPT-2 untrained GOE:     r_mean=0.5288 (structural, not training artifact)
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# ── constants ──────────────────────────────────────────────────────────────────
SEQ_LEN       = 256
N_INPUTS      = 50
MAX_DX        = 56
MIN_POS       = 32
FIT_LOW       = 3
FIT_HIGH      = 50
R2_THRESHOLD  = 0.90
SYK_NEAR_TOL  = 0.05   # |Δ - 0.25| ≤ 0.05
SEEDS         = [42, 123, 456]

OUT_DIR      = Path("research/physics/experiments/exp-049_bcft_untrained_control")
RESULTS_FILE = OUT_DIR / "results.json"

# Trained baseline for comparison
TRAINED_BASELINE = {
    "experiment":       "exp-007",
    "model":            "gpt2 (pretrained)",
    "syk_near":         44,
    "total_heads":      144,
    "delta_syk_median": 0.2493,
}


# ── lag profile ────────────────────────────────────────────────────────────────

def compute_head_attention_decay(
    attn_head: np.ndarray, min_pos: int, max_dx: int
) -> np.ndarray:
    """Mean attention weight as function of lag (exp-007 method)."""
    seq    = attn_head.shape[0]
    A      = np.zeros(max_dx, dtype=np.float64)
    counts = np.zeros(max_dx, dtype=np.float64)
    for dx in range(max_dx):
        for i in range(max(min_pos, dx), seq):
            j = i - dx
            if j >= 0:
                A[dx]      += attn_head[i, j]
                counts[dx] += 1
    mask       = counts > 0
    A[mask]   /= counts[mask]
    return A.astype(np.float32)


# ── power-law fit ──────────────────────────────────────────────────────────────

def fit_power_law(
    dx_arr: np.ndarray, y_arr: np.ndarray, low: int, high: int
) -> tuple[float, float]:
    """Return (Δ, R²). Δ = -slope/2 from log-log linear fit."""
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
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    delta  = float(-coeffs[1] / 2.0)
    return delta, r2


# ── run BCFT on one random-init model ─────────────────────────────────────────

def run_bcft_untrained(seed: int) -> dict:
    """
    Initialize GPT-2 with random seed, run exp-007 BCFT protocol,
    return per-head Δ and R² values.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"\n--- Seed {seed} ---", flush=True)
    print(f"  Initializing untrained GPT-2...", flush=True)

    config = GPT2Config()
    # attn_implementation="eager" required for output_attentions=True in transformers ≥5.8
    config._attn_implementation = "eager"
    model  = GPT2LMHeadModel(config)     # random init — no pretrained weights
    model.eval()

    n_layers   = config.n_layer    # 12
    n_heads    = config.n_head     # 12
    vocab_size = config.vocab_size
    dx_arr     = np.arange(MAX_DX)

    # Accumulators
    A_heads = {
        l: {h: np.zeros(MAX_DX, dtype=np.float64) for h in range(n_heads)}
        for l in range(n_layers)
    }

    print(f"  Running {N_INPUTS} forward passes...", flush=True)
    for inp_idx in range(N_INPUTS):
        token_ids = torch.tensor(
            rng.integers(0, vocab_size, size=(1, SEQ_LEN)), dtype=torch.long
        )
        with torch.no_grad():
            outputs = model(token_ids, output_attentions=True)

        for l in range(n_layers):
            attn = outputs.attentions[l][0].float().numpy()   # (n_heads, seq, seq)
            for h in range(n_heads):
                A_heads[l][h] += compute_head_attention_decay(attn[h], MIN_POS, MAX_DX)

        if (inp_idx + 1) % 10 == 0:
            print(f"    {inp_idx + 1}/{N_INPUTS} done", flush=True)

    # Average over inputs
    for l in range(n_layers):
        for h in range(n_heads):
            A_heads[l][h] /= N_INPUTS

    # Fit power laws
    per_head_results = []
    all_conformal: list[float] = []
    all_syk_near:  list[float] = []

    for l in range(n_layers):
        for h in range(n_heads):
            delta, r2 = fit_power_law(dx_arr, A_heads[l][h], FIT_LOW, FIT_HIGH)
            is_conformal = (r2 >= R2_THRESHOLD and not math.isnan(delta) and delta > 0)
            is_syk_near  = (is_conformal and abs(delta - 0.25) <= SYK_NEAR_TOL)

            per_head_results.append({
                "layer":       l,
                "head":        h,
                "delta":       round(delta, 6) if not math.isnan(delta) else None,
                "r2":          round(r2, 6),
                "conformal":   is_conformal,
                "syk_near":    is_syk_near,
                "a_dx0":       round(float(A_heads[l][h][0]), 6),
                "a_dx8":       round(float(A_heads[l][h][8]), 6),
                "a_dx32":      round(float(A_heads[l][h][32]), 6),
            })
            if is_conformal:
                all_conformal.append(delta)
            if is_syk_near:
                all_syk_near.append(delta)

    n_total      = n_layers * n_heads
    n_conformal  = len(all_conformal)
    n_syk_near   = len(all_syk_near)
    delta_med    = float(statistics.median(all_conformal)) if all_conformal else float("nan")
    syk_med      = float(statistics.median(all_syk_near))  if all_syk_near  else None

    # Distribution of all R² values (how many have R²>0 at all)
    r2_vals  = [h["r2"] for h in per_head_results]
    r2_above_half = sum(1 for r in r2_vals if r > 0.5)
    r2_above_zero = sum(1 for r in r2_vals if r > 0.0)

    # Δ distribution: all heads regardless of R² (to check clustering vs random)
    all_deltas_finite = [h["delta"] for h in per_head_results if h["delta"] is not None]

    print(f"  Conformal heads (R²>{R2_THRESHOLD}): {n_conformal}/{n_total}", flush=True)
    print(f"  SYK-near (|Δ-0.25|≤{SYK_NEAR_TOL}):         {n_syk_near}/{n_total}", flush=True)
    print(f"  Δ_med (conformal):              {delta_med:.4f}" if not math.isnan(delta_med)
          else f"  Δ_med: N/A (no conformal heads)", flush=True)
    print(f"  R²>0.5: {r2_above_half}/{n_total},  R²>0.0: {r2_above_zero}/{n_total}", flush=True)

    del model
    return {
        "seed":               seed,
        "n_total_heads":      n_total,
        "n_conformal":        n_conformal,
        "n_syk_near":         n_syk_near,
        "delta_median":       round(delta_med, 6) if not math.isnan(delta_med) else None,
        "syk_near_median":    round(syk_med, 6) if syk_med is not None else None,
        "r2_above_half":      r2_above_half,
        "r2_above_zero":      r2_above_zero,
        "delta_mean_all":     round(float(np.mean(all_deltas_finite)), 6)
                              if all_deltas_finite else None,
        "delta_std_all":      round(float(np.std(all_deltas_finite)), 6)
                              if all_deltas_finite else None,
        "per_head":           per_head_results,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("exp-049: BCFT conformal scaling in untrained GPT-2", flush=True)
    print(f"  Complement to exp-048 (GOE structural).", flush=True)
    print(f"  Question: does Δ ≈ 0.25 conformal structure also arise before training?", flush=True)
    print(f"  H1 (training-dependent): ~0 SYK-near heads in untrained GPT-2", flush=True)
    print(f"  H0 (structural):         ~44 SYK-near heads, comparable to trained", flush=True)
    print(f"  Protocol: SEQ_LEN={SEQ_LEN}, N_INPUTS={N_INPUTS}, FIT=[{FIT_LOW},{FIT_HIGH}], "
          f"R²>{R2_THRESHOLD}, seeds={SEEDS}", flush=True)
    print(f"  attn_implementation: eager (required for output_attentions=True, transformers ≥5.8)", flush=True)
    print(flush=True)

    seed_results = {}
    for seed in SEEDS:
        seed_results[seed] = run_bcft_untrained(seed)

    # ── cross-seed summary ─────────────────────────────────────────────────────
    syk_counts = [seed_results[s]["n_syk_near"] for s in SEEDS]
    conf_counts = [seed_results[s]["n_conformal"] for s in SEEDS]
    mean_syk   = float(np.mean(syk_counts))
    mean_conf  = float(np.mean(conf_counts))

    print("\n" + "=" * 60, flush=True)
    print("=== CROSS-SEED SUMMARY ===", flush=True)
    for s in SEEDS:
        r = seed_results[s]
        print(f"  Seed {s:4d}: SYK-near={r['n_syk_near']:3d}/{r['n_total_heads']}  "
              f"conformal={r['n_conformal']:3d}/{r['n_total_heads']}  "
              f"Δ_med={r['delta_median']}", flush=True)
    print(f"  Mean SYK-near: {mean_syk:.1f} / {seed_results[SEEDS[0]]['n_total_heads']}", flush=True)
    print(f"  Trained baseline (exp-007): {TRAINED_BASELINE['syk_near']}"
          f"/{TRAINED_BASELINE['total_heads']}  Δ_SYK={TRAINED_BASELINE['delta_syk_median']}", flush=True)

    print(flush=True)
    print("=== HYPOTHESIS VERDICTS ===", flush=True)

    # H1: training-dependent (expect ~0 SYK-near)
    # "~0" threshold: mean SYK-near < 5 (less than ~10% of trained)
    H1_THRESHOLD = 5
    h1_confirmed = mean_syk < H1_THRESHOLD
    h0_confirmed = mean_syk >= 0.5 * TRAINED_BASELINE["syk_near"]   # ≥ 50% of trained count

    h1_str = "CONFIRMED" if h1_confirmed else "NOT CONFIRMED"
    h0_str = "CONFIRMED" if h0_confirmed else "NOT CONFIRMED"

    print(f"H1 (training-dependent, ~0 SYK-near): {h1_str}", flush=True)
    print(f"   Mean SYK-near={mean_syk:.1f}, threshold=<{H1_THRESHOLD}", flush=True)
    print(f"H0 (structural, ≥50% of trained count): {h0_str}", flush=True)
    print(f"   Mean SYK-near={mean_syk:.1f}, threshold≥{0.5 * TRAINED_BASELINE['syk_near']}", flush=True)

    print(flush=True)
    print("=== PHYSICAL INTERPRETATION ===", flush=True)
    if h1_confirmed:
        print("Conformal position-space structure IS training-dependent.", flush=True)
        print("Random attention (near-uniform from small init) does not exhibit Δ≈0.25 power law.", flush=True)
        print("Two-layer physical picture:", flush=True)
        print("  Layer 1 (structural): GOE weight-space chaos from Gaussian init + product structure.", flush=True)
        print("  Layer 2 (learned):    Conformal Δ≈0.25 position-space structure from gradient descent.", flush=True)
        print("Training adds conformal dynamics on top of a pre-existing chaotic weight background.", flush=True)
    elif h0_confirmed:
        print("Conformal position-space structure is STRUCTURAL (arises without training).", flush=True)
        print("Both GOE (weight-space) and conformal (position-space) structure are initialization-level.", flush=True)
        print("This is surprising — random attention should tend to uniform, not power-law.", flush=True)
        print("Further investigation needed: what mechanism produces power-law in untrained attention?", flush=True)
    else:
        print(f"Intermediate result: mean SYK-near={mean_syk:.1f}.", flush=True)
        print("Some conformal signal present but substantially less than trained baseline.", flush=True)
        print("Partial training-dependence: initialization seeds weak structure, training amplifies it.", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────
    result = {
        "experiment": "exp-049",
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H1": ("Untrained GPT-2 shows ~0 SYK-near heads. Conformal position-space "
                   "structure requires training. Near-uniform attention from small init "
                   "(σ=0.02) gives no power-law decay."),
            "H0": ("Untrained GPT-2 shows substantial SYK-near count (≥50% of trained). "
                   "Conformal structure is also structural — arises from architecture + init."),
        },
        "protocol": {
            "model":            "gpt2 (random init, GPT2LMHeadModel(GPT2Config()))",
            "architecture":     "gpt2 (12L, 12H, 64d_k, 768 hidden)",
            "init":             "default PyTorch random init, std=0.02 (config._attn_implementation='eager')",
            "seq_len":          SEQ_LEN,
            "n_inputs":         N_INPUTS,
            "fit_range":        [FIT_LOW, FIT_HIGH],
            "R2_threshold":     R2_THRESHOLD,
            "syk_near_tol":     SYK_NEAR_TOL,
            "seeds":            SEEDS,
            "note":             "Same BCFT protocol as exp-007/exp-043. No pretrained weights.",
        },
        "reference": {
            "exp007_gpt2_trained_softmax": {
                "syk_near": 44, "total_heads": 144, "delta_syk_median": 0.2493,
            },
            "exp043_gpt2_trained_normsig": {
                "syk_near": 13, "total_heads": 144, "delta_syk_median": 0.234,
            },
            "exp048_gpt2_untrained_goe": {
                "r_mean": 0.5288, "verdict": "GOE-like (structural)",
            },
        },
        "seeds":   {str(s): seed_results[s] for s in SEEDS},
        "summary": {
            "mean_syk_near":        round(mean_syk, 2),
            "mean_conformal":       round(mean_conf, 2),
            "seed_syk_counts":      {str(s): seed_results[s]["n_syk_near"] for s in SEEDS},
            "seed_conformal_counts":{str(s): seed_results[s]["n_conformal"] for s in SEEDS},
            "trained_syk_baseline": TRAINED_BASELINE["syk_near"],
            "total_heads":          seed_results[SEEDS[0]]["n_total_heads"],
        },
        "verdicts": {
            "H1_training_dependent": h1_confirmed,
            "H0_structural":         h0_confirmed,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
