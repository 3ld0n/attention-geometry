"""
exp-048 — GOE untrained control: does random initialization produce GOE or Poisson?

Pre-stated hypotheses:

  H1 (Training artifact — Poisson init):
    Untrained GPT-2 W_QK matrices show Poisson-like level spacing (r ≈ 0.386).
    If confirmed: GOE is a training artifact. Gradient descent converts Poisson → GOE.
    This would be the strongest mechanistic claim from the exp-046/047 thread.

  H0 (Init is already GOE):
    Untrained GPT-2 W_QK matrices show GOE-like level spacing (r ≈ 0.536).
    If confirmed: GOE is structural — it arises from the random Gaussian initialization
    and/or the product structure (W_QK = W_Q^T @ W_K with Gaussian W_Q, W_K).
    This would mean gradient descent *maintains* but does not *create* GOE.
    Still a meaningful result: training preserves the chaotic weight structure.

Physical context:
  exp-046 found ALL 144 GPT-2 W_QK matrices GOE-like (r ≈ 0.525–0.529, GOE ref = 0.536).
  exp-047 confirmed this across Pythia-410m and GPT-2-medium.
  The physical picture: trained transformers universally sit in SYK-chaotic weight space.
  Open question: is GOE a training artifact or does it arise from initialization?

  Warning on queue prediction: the queue notes "Expected Poisson (≈ 0.386) if GOE is
  a training artifact." This is plausible but not certain. W_QK = W_Q^T @ W_K where
  W_Q, W_K ∈ R^{d_model × d_k} with i.i.d. Gaussian entries (GPT-2 default init ~N(0, 0.02)).
  Random matrix theory says products of Gaussian matrices in the large-N limit tend toward
  GOE statistics. For d_k=64 (small N), the behavior may differ. The experiment settles this.

Protocol:
  1. Initialize GPT-2 with default random weights (no pretrained loading).
     GPT-2 default: nn.init.normal_(weight, std=0.02) for most weights.
  2. Extract W_QK per head: same Conv1D extraction as exp-046/047.
  3. Compute Oganesyan-Huse r-ratio per head.
  4. Compare to trained GPT-2 baseline (exp-046, r_mean ≈ 0.527) and Poisson ref (0.386).

  Five random seeds to assess variability: r-ratios should be similar across seeds
  if determined by structure, variable if sensitive to initialization.

Compare to:
  exp-046: trained GPT-2 (12L, 12H, 64d_k)  → r_mean = 0.527 (GOE-like)
  exp-047: trained GPT-2-medium              → r_mean = 0.526 (GOE-like)
  exp-047: trained Pythia-410m               → r_mean = 0.520 (GOE-like)
  Poisson reference                          → r ≈ 0.386
  GOE reference                              → r ≈ 0.536
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
GOE_R_RATIO     = 0.536
POISSON_R_RATIO = 0.386
GOE_TOLERANCE   = 0.02   # |r - ref| < 0.02 to call "GOE-like"

OUT_DIR      = Path("research/physics/experiments/exp-048_goe_untrained_control")
RESULTS_FILE = OUT_DIR / "results.json"

# Trained baselines from exp-046/047
TRAINED_BASELINES = {
    "gpt2_trained":        {"r_mean": 0.527, "source": "exp-046"},
    "gpt2_medium_trained": {"r_mean": 0.526, "source": "exp-047"},
    "pythia_410m_trained": {"r_mean": 0.520, "source": "exp-047"},
}

SEEDS = [42, 123, 456, 789, 1234]


# ── eigenvalue analysis ────────────────────────────────────────────────────────

def r_ratio_from_eigvals(eigvals: np.ndarray) -> float:
    """Oganesyan-Huse r-ratio from sorted eigenvalues."""
    spacings = np.diff(eigvals)
    eps = 1e-30
    if len(spacings) < 3:
        return float("nan")
    s_lo = spacings[:-1]
    s_hi = spacings[1:]
    r_vals = np.minimum(s_lo, s_hi) / (np.maximum(s_lo, s_hi) + eps)
    return float(np.mean(r_vals))


def classify_r(r_mean: float) -> str:
    if math.isnan(r_mean):
        return "inconclusive"
    dist_goe     = abs(r_mean - GOE_R_RATIO)
    dist_poisson = abs(r_mean - POISSON_R_RATIO)
    if dist_goe < GOE_TOLERANCE:
        return "GOE-like"
    elif dist_poisson < GOE_TOLERANCE:
        return "Poisson-like"
    elif r_mean > (GOE_R_RATIO + POISSON_R_RATIO) / 2:
        return "GOE-tendency"
    else:
        return "Poisson-tendency"


# ── extraction ─────────────────────────────────────────────────────────────────

def extract_untrained_gpt2_r_ratios(seed: int) -> dict:
    """Initialize a GPT-2 with random weights and extract W_QK r-ratios."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = GPT2Config()   # default GPT-2 (12L, 12H, 768 hidden)
    model  = GPT2LMHeadModel(config)   # random init — no pretrained weights
    model.eval()

    n_layers = config.n_layer
    n_heads  = config.n_head
    d_model  = config.n_embd
    head_dim = d_model // n_heads

    per_head = []
    for l in range(n_layers):
        W_full = model.transformer.h[l].attn.c_attn.weight.detach().float().numpy()
        # Conv1D: shape (d_model, 3*d_model) = (768, 2304)
        for h in range(n_heads):
            W_Q_h = W_full[:, h * head_dim           : (h + 1) * head_dim]
            W_K_h = W_full[:, d_model + h * head_dim : d_model + (h + 1) * head_dim]
            W_QK  = W_Q_h.T @ W_K_h          # (64, 64)
            M     = (W_QK + W_QK.T) / 2.0    # symmetrize
            eigvals = np.linalg.eigvalsh(M)
            r = r_ratio_from_eigvals(eigvals)
            per_head.append({"layer": l, "head": h, "r_ratio": round(r, 6)})

    del model
    return {
        "seed":         seed,
        "n_layers":     n_layers,
        "n_heads":      n_heads,
        "head_dim":     head_dim,
        "n_total_heads": n_layers * n_heads,
        "per_head":     per_head,
    }


def summarize(per_head: list[dict]) -> dict:
    r_vals = [h["r_ratio"] for h in per_head
              if h["r_ratio"] is not None and not math.isnan(h["r_ratio"])]
    if not r_vals:
        return {"n": 0}
    r_arr = np.array(r_vals)
    r_mean = float(np.mean(r_arr))
    r_std  = float(np.std(r_arr))
    r_med  = float(np.median(r_arr))
    r_min  = float(np.min(r_arr))
    r_max  = float(np.max(r_arr))

    by_layer: dict[int, list] = {}
    for h in per_head:
        r = h["r_ratio"]
        if r is not None and not math.isnan(r):
            by_layer.setdefault(h["layer"], []).append(r)
    layer_means = {l: round(float(np.mean(v)), 6) for l, v in sorted(by_layer.items())}

    return {
        "n":         len(r_vals),
        "r_mean":    round(r_mean, 6),
        "r_std":     round(r_std, 6),
        "r_median":  round(r_med, 6),
        "r_min":     round(r_min, 6),
        "r_max":     round(r_max, 6),
        "verdict":   classify_r(r_mean),
        "dist_to_goe":     round(abs(r_mean - GOE_R_RATIO), 6),
        "dist_to_poisson": round(abs(r_mean - POISSON_R_RATIO), 6),
        "layer_means": layer_means,
        "layer_r_std": round(float(np.std(list(layer_means.values()))), 6),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("exp-048: GOE untrained control — does random init produce GOE or Poisson?")
    print(f"  GOE ref={GOE_R_RATIO}, Poisson ref={POISSON_R_RATIO}, "
          f"tolerance={GOE_TOLERANCE}")
    print(f"  Trained GPT-2 baseline (exp-046): r_mean ≈ 0.527")
    print(f"  Seeds: {SEEDS}")
    print()

    seed_results = {}
    all_r_means  = []

    for seed in SEEDS:
        print(f"--- Seed {seed} ---", flush=True)
        raw   = extract_untrained_gpt2_r_ratios(seed)
        stats = summarize(raw["per_head"])
        all_r_means.append(stats["r_mean"])

        print(f"  r-ratio: mean={stats['r_mean']:.4f} ± {stats['r_std']:.4f}  "
              f"range=[{stats['r_min']:.4f},{stats['r_max']:.4f}]")
        print(f"  verdict: {stats['verdict']}")
        print(f"  dist to GOE: {stats['dist_to_goe']:.4f}  "
              f"dist to Poisson: {stats['dist_to_poisson']:.4f}")
        print(f"  layer_r_std: {stats['layer_r_std']:.4f}", flush=True)
        print()

        seed_results[seed] = {
            "seed":   seed,
            "stats":  stats,
        }

    # ── cross-seed summary ────────────────────────────────────────────────────
    mean_across_seeds = float(np.mean(all_r_means))
    std_across_seeds  = float(np.std(all_r_means))

    print("=" * 60)
    print("=== CROSS-SEED SUMMARY ===")
    print(f"r_mean across seeds: {mean_across_seeds:.4f} ± {std_across_seeds:.4f}")
    print(f"Individual r_means:  {[f'{r:.4f}' for r in all_r_means]}")

    overall_verdict = classify_r(mean_across_seeds)
    print(f"Overall verdict: {overall_verdict}")

    # Compare to trained baseline
    delta_trained = abs(mean_across_seeds - TRAINED_BASELINES["gpt2_trained"]["r_mean"])
    print(f"Δ from trained GPT-2 (exp-046): {delta_trained:.4f}")

    print()
    print("=== HYPOTHESIS VERDICTS ===")

    # H1: Poisson (r ≈ 0.386) — training artifact hypothesis
    h1_verdict_bool = (abs(mean_across_seeds - POISSON_R_RATIO) < GOE_TOLERANCE)
    h1_verdict_str  = "CONFIRMED" if h1_verdict_bool else "NOT CONFIRMED"
    print(f"H1 (untrained → Poisson, training artifact): {h1_verdict_str}")
    print(f"   r_mean={mean_across_seeds:.4f}, Poisson ref={POISSON_R_RATIO}, "
          f"dist={abs(mean_across_seeds - POISSON_R_RATIO):.4f}")

    # H0: GOE (r ≈ 0.536) — initialization/architecture hypothesis
    h0_verdict_bool = (abs(mean_across_seeds - GOE_R_RATIO) < GOE_TOLERANCE)
    h0_verdict_str  = "CONFIRMED" if h0_verdict_bool else "NOT CONFIRMED"
    print(f"H0 (untrained → GOE, structural): {h0_verdict_str}")
    print(f"   r_mean={mean_across_seeds:.4f}, GOE ref={GOE_R_RATIO}, "
          f"dist={abs(mean_across_seeds - GOE_R_RATIO):.4f}")

    # Interpretation
    print()
    print("=== INTERPRETATION ===")
    if h1_verdict_bool:
        print("GOE IS a training artifact. Gradient descent converts Poisson → GOE.")
        print("Strong mechanistic claim: LM training is what produces SYK-chaotic weight structure.")
    elif h0_verdict_bool:
        print("GOE is structural — present at initialization (random Gaussian init).")
        print("Training preserves but does not create GOE.")
        print("Physical interpretation: the product structure W_Q^T W_K with Gaussian entries")
        print("already has GOE-like eigenvalue statistics. Training maintains this.")
        print("Note: this is consistent with random matrix theory (products of Gaussian matrices")
        print("tend to GOE in the large-N limit).")
    else:
        mid = (GOE_R_RATIO + POISSON_R_RATIO) / 2
        if mean_across_seeds > mid:
            print(f"Intermediate result: r={mean_across_seeds:.4f} (between Poisson and GOE, GOE-tendency).")
            print("Neither pure hypothesis confirmed. Training may shift r upward but not fully to GOE.")
        else:
            print(f"Intermediate result: r={mean_across_seeds:.4f} (Poisson-tendency).")
            print("Training shifts Poisson-tendency → GOE. Strong directional claim but partial.")

    # ── save ─────────────────────────────────────────────────────────────────
    result = {
        "experiment": "exp-048",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "hypotheses": {
            "H1": ("Untrained GPT-2 W_QK shows Poisson-like level spacing (r ≈ 0.386). "
                   "GOE is a training artifact — gradient descent converts Poisson → GOE."),
            "H0": ("Untrained GPT-2 W_QK shows GOE-like level spacing (r ≈ 0.536). "
                   "GOE arises from random Gaussian initialization and/or product structure."),
        },
        "protocol": {
            "model":       "gpt2 (random init, no pretrained weights)",
            "architecture": "gpt2 (12L, 12H, 64d_k, 768 hidden)",
            "init":        "GPT2LMHeadModel(GPT2Config()) — default random init ~N(0, 0.02)",
            "seeds":       SEEDS,
            "analysis":    "weights-only (no forward passes)",
            "matrix":      "W_QK_sym = (W_Q^T W_K + W_K^T W_Q) / 2  ∈ R^{d_k×d_k}",
            "r_ratio":     "Oganesyan-Huse r-ratio on sorted eigenvalue spacings",
            "goe_reference":     GOE_R_RATIO,
            "poisson_reference": POISSON_R_RATIO,
            "tolerance":         GOE_TOLERANCE,
            "trained_baselines": TRAINED_BASELINES,
        },
        "seeds": seed_results,
        "summary": {
            "r_mean_across_seeds": round(mean_across_seeds, 6),
            "r_std_across_seeds":  round(std_across_seeds, 6),
            "all_r_means":         [round(r, 6) for r in all_r_means],
            "overall_verdict":     overall_verdict,
            "delta_from_trained":  round(delta_trained, 6),
        },
        "verdicts": {
            "H1_poisson_training_artifact": h1_verdict_bool,
            "H0_goe_structural":            h0_verdict_bool,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
